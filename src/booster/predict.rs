//! Prediction Methods
//!
//! Prediction, probability, contribution, and feature importance methods for the booster.

use crate::MultiOutputBooster;

use crate::booster::config::{CalibrationMethod, ContributionsMethod};
use crate::data::ColumnarMatrix;
use crate::objective::Objective;
use crate::{Matrix, PerpetualBooster, shapley::predict_contributions_row_shapley, tree::core::Tree, utils::odds};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

impl PerpetualBooster {
    /// Generate predictions for the given data.
    ///
    /// # Arguments
    ///
    /// * `data` - The feature matrix.
    /// * `parallel` - If `true`, predictions are computed in parallel using Rayon.
    ///
    /// # Returns
    ///
    /// A vector of predicted values.
    /// * For regression/ranking: The raw predicted score.
    /// * For classification: The log-odds (logit). Apply sigmoid to get probabilities.
    pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let mut init_preds = vec![self.base_score; data.rows];
        let trees = self.get_prediction_trees();
        trees.iter().for_each(|tree| {
            for (p_, val) in init_preds
                .iter_mut()
                .zip(tree.predict(data, parallel, &self.cfg.missing))
            {
                *p_ += val;
            }
        });
        init_preds
    }

    /// Generate predictions for columnar data (Zero-Copy).
    ///
    /// # Arguments
    ///
    /// * `data` - The columnar feature matrix.
    /// * `parallel` - If `true`, predictions are computed in parallel.
    pub fn predict_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool) -> Vec<f64> {
        let mut init_preds = vec![self.base_score; data.rows];
        self.get_prediction_trees().iter().for_each(|tree| {
            for (p_, val) in init_preds
                .iter_mut()
                .zip(tree.predict_columnar(data, parallel, &self.cfg.missing))
            {
                *p_ += val;
            }
        });
        init_preds
    }

    /// Generate class probabilities (sigmoid of log-odds) for the given data.
    ///
    /// Only meaningful for binary classification with `LogLoss` objective.
    ///
    /// * `data` - The feature matrix.
    /// * `parallel` - Predict in parallel.
    pub fn predict_proba(&self, data: &Matrix<f64>, parallel: bool, calibrated: bool) -> Vec<f64> {
        let preds = self.predict(data, parallel);
        let mut proba: Vec<f64> = if parallel {
            preds.par_iter().map(|p| 1.0 / (1.0 + (-p).exp())).collect()
        } else {
            preds.iter().map(|p| 1.0 / (1.0 + (-p).exp())).collect()
        };

        if let Some(calibrator) = self.isotonic_calibrator.as_ref().filter(|_| calibrated) {
            let scores = match self.cfg.calibration_method {
                CalibrationMethod::Conformal => proba.clone(),
                _ => self.get_calibration_scores(data, parallel),
            };
            proba = calibrator.transform(&scores);
        }
        proba
    }

    /// Generate class probabilities for columnar data (zero-copy).
    ///
    /// * `data` - The columnar feature matrix.
    /// * `parallel` - Predict in parallel.
    pub fn predict_proba_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool, calibrated: bool) -> Vec<f64> {
        let preds = self.predict_columnar(data, parallel);
        let mut proba: Vec<f64> = if parallel {
            preds.par_iter().map(|p| 1.0 / (1.0 + (-p).exp())).collect()
        } else {
            preds.iter().map(|p| 1.0 / (1.0 + (-p).exp())).collect()
        };

        if let Some(calibrator) = self.isotonic_calibrator.as_ref().filter(|_| calibrated) {
            let scores = match self.cfg.calibration_method {
                CalibrationMethod::Conformal => proba.clone(),
                _ => self.get_calibration_scores_columnar(data, parallel),
            };
            proba = calibrator.transform(&scores);
        }
        proba
    }

    /// Calculate Feature Contributions (SHAP-like values).
    ///
    /// Computes how much each feature contributed to the prediction for each sample.
    /// - The sum of contributions + bias equals the prediction.
    /// - The last column in the returned matrix is the bias term.
    ///
    /// # Arguments
    ///
    /// * `data` - Feature matrix.
    /// * `method` - Method to calculate contributions (`Average` is the standard TreeSHAP approximation).
    /// * `parallel` - Run in parallel.
    pub fn predict_contributions(&self, data: &Matrix<f64>, method: ContributionsMethod, parallel: bool) -> Vec<f64> {
        match method {
            ContributionsMethod::Average => self.predict_contributions_average(data, parallel),
            ContributionsMethod::ProbabilityChange => {
                match self.cfg.objective {
                    Objective::LogLoss => {}
                    _ => panic!("ProbabilityChange contributions method is only valid when LogLoss objective is used."),
                }
                self.predict_contributions_probability_change(data, parallel)
            }
            _ => self.predict_contributions_tree_alone(data, parallel, method),
        }
    }

    // All of the contribution calculation methods, except for average are calculated
    // using just the model, so we don't need to have separate methods, we can instead
    // just have this one method, that dispatches to each one respectively.
    fn predict_contributions_tree_alone(
        &self,
        data: &Matrix<f64>,
        parallel: bool,
        method: ContributionsMethod,
    ) -> Vec<f64> {
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];

        // Add the bias term to every bias value...
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += self.base_score);

        let row_pred_fn = match method {
            ContributionsMethod::Weight => Tree::predict_contributions_row_weight,
            ContributionsMethod::BranchDifference => Tree::predict_contributions_row_branch_difference,
            ContributionsMethod::MidpointDifference => Tree::predict_contributions_row_midpoint_difference,
            ContributionsMethod::ModeDifference => Tree::predict_contributions_row_mode_difference,
            ContributionsMethod::Shapley => predict_contributions_row_shapley,
            ContributionsMethod::Average | ContributionsMethod::ProbabilityChange => unreachable!(),
        };
        // Clean this up..
        // materializing a row, and then passing that to all of the
        // trees seems to be the fastest approach (5X faster), we should test
        // something like this for normal predictions.
        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().for_each(|t| {
                        row_pred_fn(t, &r_, c, &self.cfg.missing);
                    });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().for_each(|t| {
                        row_pred_fn(t, &r_, c, &self.cfg.missing);
                    });
                });
        }

        contribs
    }

    /// Generate predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    fn predict_contributions_average(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let weights: Vec<HashMap<usize, f64>> = if parallel {
            self.get_prediction_trees()
                .par_iter()
                .map(|t| t.distribute_leaf_weights())
                .collect()
        } else {
            self.get_prediction_trees()
                .iter()
                .map(|t| t.distribute_leaf_weights())
                .collect()
        };
        let mut contribs = vec![0.0; (data.cols + 1) * data.rows];

        // Add the bias term to every bias value...
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += self.base_score);

        // Clean this up..
        // materializing a row, and then passing that to all of the
        // trees seems to be the fastest approach (5X faster), we should test
        // something like this for normal predictions.
        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .zip(weights.iter())
                        .for_each(|(t, w)| {
                            t.predict_contributions_row_average(&r_, c, w, &self.cfg.missing);
                        });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .zip(weights.iter())
                        .for_each(|(t, w)| {
                            t.predict_contributions_row_average(&r_, c, w, &self.cfg.missing);
                        });
                });
        }

        contribs
    }

    fn predict_contributions_probability_change(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += odds(self.base_score));

        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().fold(self.base_score, |acc, t| {
                        t.predict_contributions_row_probability_change(&r_, c, &self.cfg.missing, acc)
                    });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().fold(self.base_score, |acc, t| {
                        t.predict_contributions_row_probability_change(&r_, c, &self.cfg.missing, acc)
                    });
                });
        }
        contribs
    }

    /// Generate node predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict_nodes(&self, data: &Matrix<f64>, parallel: bool) -> Vec<Vec<HashSet<usize>>> {
        let mut v = Vec::with_capacity(data.rows);
        self.get_prediction_trees().iter().for_each(|tree| {
            let tree_nodes = tree.predict_nodes(data, parallel, &self.cfg.missing);
            v.push(tree_nodes);
        });
        v
    }

    /// Generate node predictions on columnar data using the gradient booster (zero-copy from Polars).
    pub fn predict_nodes_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool) -> Vec<Vec<HashSet<usize>>> {
        let mut v = Vec::with_capacity(data.rows);
        self.get_prediction_trees().iter().for_each(|tree| {
            let tree_nodes = tree.predict_nodes_columnar(data, parallel, &self.cfg.missing);
            v.push(tree_nodes);
        });
        v
    }

    fn predict_contributions_average_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool) -> Vec<f64> {
        let weights: Vec<HashMap<usize, f64>> = if parallel {
            self.get_prediction_trees()
                .par_iter()
                .map(|t| t.distribute_leaf_weights())
                .collect()
        } else {
            self.get_prediction_trees()
                .iter()
                .map(|t| t.distribute_leaf_weights())
                .collect()
        };
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += self.base_score);

        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .zip(weights.iter())
                        .for_each(|(t, w)| {
                            t.predict_contributions_row_average(&r_, c, w, &self.cfg.missing);
                        });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .zip(weights.iter())
                        .for_each(|(t, w)| {
                            t.predict_contributions_row_average(&r_, c, w, &self.cfg.missing);
                        });
                });
        }

        contribs
    }

    fn predict_contributions_probability_change_columnar(
        &self,
        data: &ColumnarMatrix<f64>,
        parallel: bool,
    ) -> Vec<f64> {
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += odds(self.base_score));

        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().fold(self.base_score, |acc, t| {
                        t.predict_contributions_row_probability_change(&r_, c, &self.cfg.missing, acc)
                    });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().fold(self.base_score, |acc, t| {
                        t.predict_contributions_row_probability_change(&r_, c, &self.cfg.missing, acc)
                    });
                });
        }
        contribs
    }

    fn predict_contributions_tree_alone_columnar(
        &self,
        data: &ColumnarMatrix<f64>,
        parallel: bool,
        method: ContributionsMethod,
    ) -> Vec<f64> {
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        let row_pred_fn = match method {
            ContributionsMethod::Weight => Tree::predict_contributions_row_weight,
            ContributionsMethod::BranchDifference => Tree::predict_contributions_row_branch_difference,
            ContributionsMethod::MidpointDifference => Tree::predict_contributions_row_midpoint_difference,
            ContributionsMethod::ModeDifference => Tree::predict_contributions_row_mode_difference,
            ContributionsMethod::Shapley => predict_contributions_row_shapley,
            ContributionsMethod::Average | ContributionsMethod::ProbabilityChange => unreachable!(),
        };
        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().for_each(|t| {
                        row_pred_fn(t, &r_, c, &self.cfg.missing);
                    });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().for_each(|t| {
                        row_pred_fn(t, &r_, c, &self.cfg.missing);
                    });
                });
        }
        contribs
    }

    pub fn predict_contributions_columnar(
        &self,
        data: &ColumnarMatrix<f64>,
        method: ContributionsMethod,
        parallel: bool,
    ) -> Vec<f64> {
        match method {
            ContributionsMethod::Average => self.predict_contributions_average_columnar(data, parallel),
            ContributionsMethod::ProbabilityChange => {
                match self.cfg.objective {
                    Objective::LogLoss => {}
                    _ => panic!("ProbabilityChange contributions method is only valid when LogLoss objective is used."),
                }
                self.predict_contributions_probability_change_columnar(data, parallel)
            }
            _ => self.predict_contributions_tree_alone_columnar(data, parallel, method),
        }
    }
}

impl MultiOutputBooster {
    /// Generate predictions on data using the multi-output booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        self.boosters
            .iter()
            .flat_map(|b| b.predict(data, parallel))
            .collect::<Vec<f64>>()
    }

    /// Generate predictions on columnar data using the multi-output booster (zero-copy from Polars).
    pub fn predict_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool) -> Vec<f64> {
        self.boosters
            .iter()
            .flat_map(|b| b.predict_columnar(data, parallel))
            .collect::<Vec<f64>>()
    }

    /// Generate probabilities on data using the multi-output booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict_proba(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let log_odds = self.predict(data, parallel);
        let data_log_odds = Matrix::new(&log_odds, data.rows, self.n_boosters);
        let mut preds = Vec::with_capacity(log_odds.len());
        for row in 0..data.rows {
            let row_values = data_log_odds.get_row(row);
            let max_val = row_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let y_p_exp = row_values.iter().map(|e| (e - max_val).exp()).collect::<Vec<f64>>();
            let y_p_exp_sum = y_p_exp.iter().sum::<f64>();
            let probabilities = y_p_exp.iter().map(|e| e / y_p_exp_sum).collect::<Vec<f64>>();
            preds.extend(probabilities);
        }
        preds
    }

    /// Generate probabilities on columnar data using the multi-output booster (zero-copy from Polars).
    pub fn predict_proba_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool) -> Vec<f64> {
        let log_odds = self.predict_columnar(data, parallel);
        let data_log_odds = Matrix::new(&log_odds, data.rows, self.n_boosters);
        let mut preds = Vec::with_capacity(log_odds.len());
        for row in 0..data.rows {
            let row_values = data_log_odds.get_row(row);
            let max_val = row_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let y_p_exp = row_values.iter().map(|e| (e - max_val).exp()).collect::<Vec<f64>>();
            let y_p_exp_sum = y_p_exp.iter().sum::<f64>();
            let probabilities = y_p_exp.iter().map(|e| e / y_p_exp_sum).collect::<Vec<f64>>();
            preds.extend(probabilities);
        }
        preds
    }

    /// Generate node predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict_nodes(&self, data: &Matrix<f64>, parallel: bool) -> Vec<Vec<Vec<HashSet<usize>>>> {
        self.boosters.iter().map(|b| b.predict_nodes(data, parallel)).collect()
    }

    /// Generate node predictions on columnar data using the gradient booster (zero-copy from Polars).
    pub fn predict_nodes_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool) -> Vec<Vec<Vec<HashSet<usize>>>> {
        self.boosters
            .iter()
            .map(|b| b.predict_nodes_columnar(data, parallel))
            .collect()
    }

    /// Predict with the fitted booster on new data, returning the feature
    /// contribution matrix. The last column is the bias term.
    pub fn predict_contributions(&self, data: &Matrix<f64>, method: ContributionsMethod, parallel: bool) -> Vec<f64> {
        self.boosters
            .iter()
            .flat_map(|b| b.predict_contributions(data, method, parallel))
            .collect::<Vec<f64>>()
    }

    /// Calculate Feature Contributions for columnar data.
    pub fn predict_contributions_columnar(
        &self,
        data: &ColumnarMatrix<f64>,
        method: ContributionsMethod,
        parallel: bool,
    ) -> Vec<f64> {
        self.boosters
            .iter()
            .flat_map(|b| b.predict_contributions_columnar(data, method, parallel))
            .collect::<Vec<f64>>()
    }

    pub fn predict_intervals(&self, data: &Matrix<f64>, parallel: bool) -> HashMap<String, Vec<Vec<f64>>> {
        let mut results: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
        for booster in &self.boosters {
            let booster_intervals = booster.predict_intervals(data, parallel);
            for (alpha, intervals) in booster_intervals {
                let entry = results.entry(alpha).or_insert_with(|| vec![Vec::new(); data.rows]);
                for (i, sample_interval) in intervals.into_iter().enumerate() {
                    entry[i].extend(sample_interval);
                }
            }
        }
        results
    }

    pub fn predict_intervals_columnar(
        &self,
        data: &ColumnarMatrix<f64>,
        parallel: bool,
    ) -> HashMap<String, Vec<Vec<f64>>> {
        let mut results: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
        let n_samples = data.index.len();
        for booster in &self.boosters {
            let booster_intervals = booster.predict_intervals_columnar(data, parallel);
            for (alpha, intervals) in booster_intervals {
                let entry = results.entry(alpha).or_insert_with(|| vec![Vec::new(); n_samples]);
                for (i, sample_interval) in intervals.into_iter().enumerate() {
                    entry[i].extend(sample_interval);
                }
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::objective::Objective;
    use crate::tree::core::Tree;
    use approx::assert_relative_eq;

    fn create_mock_booster() -> PerpetualBooster {
        let mut booster = PerpetualBooster::default();
        booster.cfg.objective = Objective::SquaredLoss;
        booster.base_score = 0.5;
        let mut tree = Tree::new();
        let root = Node {
            num: 0,
            weight_value: 0.1,
            hessian_sum: 10.0,
            split_value: 0.0,
            split_feature: 0,
            split_gain: 0.0,
            missing_node: 0,
            left_child: 0,
            right_child: 0,
            is_leaf: true,
            parent_node: 0,
            left_cats: None,
            stats: None,
        };
        tree.nodes.insert(0, root);
        tree.n_leaves = 1;
        booster.trees = vec![tree];
        booster
    }

    fn create_mock_multi_booster() -> MultiOutputBooster {
        let mut booster = MultiOutputBooster::default();
        booster.boosters = vec![create_mock_booster(), create_mock_booster()];
        booster.n_boosters = 2;
        booster
    }

    #[test]
    fn test_predict_basic() {
        let booster = create_mock_booster();
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let preds = booster.predict(&data, false);
        assert_eq!(preds.len(), 1);
        assert_relative_eq!(preds[0], 0.6, epsilon = 1e-7);
    }

    #[test]
    fn test_predict_parallel() {
        let booster = create_mock_booster();
        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let preds = booster.predict(&data, true);
        assert_eq!(preds.len(), 2);
        assert_relative_eq!(preds[0], 0.6, epsilon = 1e-7);
    }

    #[test]
    fn test_predict_columnar() {
        let booster = create_mock_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let preds = booster.predict_columnar(&data, false);
        assert_eq!(preds.len(), 2);
        assert_relative_eq!(preds[0], 0.6, epsilon = 1e-7);
    }

    #[test]
    fn test_predict_proba_basic() {
        let mut booster = create_mock_booster();
        booster.cfg.objective = Objective::LogLoss;
        booster.base_score = 0.0;
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let proba = booster.predict_proba(&data, false, false);
        assert_relative_eq!(proba[0], 0.52497918747894, epsilon = 1e-7);
    }

    #[test]
    fn test_predict_proba_parallel() {
        let mut booster = create_mock_booster();
        booster.cfg.objective = Objective::LogLoss;
        booster.base_score = 0.0;
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let proba = booster.predict_proba(&data, true, false);
        assert_relative_eq!(proba[0], 0.52497918747894, epsilon = 1e-7);
    }

    #[test]
    fn test_predict_contributions_methods() {
        let booster = create_mock_booster();
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let methods = vec![
            ContributionsMethod::Weight,
            ContributionsMethod::Average,
            ContributionsMethod::MidpointDifference,
            ContributionsMethod::BranchDifference,
            ContributionsMethod::ModeDifference,
            ContributionsMethod::Shapley,
        ];
        for method in methods {
            let contribs = booster.predict_contributions(&data, method, false);
            assert_eq!(contribs.len(), 3);
        }
    }

    #[test]
    fn test_predict_contributions_probability_change() {
        let mut booster = create_mock_booster();
        booster.cfg.objective = Objective::LogLoss;
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::ProbabilityChange, false);
        assert_eq!(contribs.len(), 3);
        // Sum of contributions + bias should ≈ probability?
        // Actually for ProbabilityChange it depends on the implementation.
    }

    #[test]
    fn test_predict_proba_calibrated() {
        let mut booster = create_mock_booster();
        booster.cfg.objective = Objective::LogLoss;
        // Adding a dummy calibrator would be complex, but let's see if we can trigger the code path.
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let proba = booster.predict_proba(&data, false, true);
        assert_eq!(proba.len(), 1);
    }

    #[test]
    fn test_predict_nodes() {
        let booster = create_mock_booster();
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let nodes = booster.predict_nodes(&data, false);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].len(), 1);
        assert!(nodes[0][0].contains(&0));
    }

    #[test]
    fn test_multi_output_predict_basic() {
        let booster = create_mock_multi_booster();
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let preds = booster.predict(&data, false);
        assert_eq!(preds.len(), 2);
        assert_relative_eq!(preds[0], 0.6, epsilon = 1e-7);
    }

    #[test]
    fn test_multi_output_predict_proba_basic() {
        let booster = create_mock_multi_booster();
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let proba = booster.predict_proba(&data, false);
        assert_eq!(proba.len(), 2);
        assert_relative_eq!(proba[0], 0.5, epsilon = 1e-7);
    }

    #[test]
    fn test_multi_output_predict_intervals_columnar() {
        let booster = create_mock_multi_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let intervals = booster.predict_intervals_columnar(&data, false);
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_predict_contributions_columnar() {
        let booster = create_mock_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let contribs = booster.predict_contributions_columnar(&data, ContributionsMethod::Weight, false);
        assert_eq!(contribs.len(), 6); // 2 rows * (2 features + 1 bias)
    }

    #[test]
    fn test_multi_output_predict_contributions_columnar() {
        let booster = create_mock_multi_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let contribs = booster.predict_contributions_columnar(&data, ContributionsMethod::Weight, false);
        assert_eq!(contribs.len(), 12); // 2 rows * 3 items * 2 boosters
    }

    // ---- Additional tests for uncovered paths ----

    #[test]
    fn test_predict_proba_columnar() {
        let mut booster = create_mock_booster();
        booster.cfg.objective = Objective::LogLoss;
        booster.base_score = 0.0;
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let proba = booster.predict_proba_columnar(&data, false, false);
        assert_eq!(proba.len(), 2);
        for p in &proba {
            assert!(*p > 0.0 && *p < 1.0);
        }
    }

    #[test]
    fn test_predict_proba_columnar_parallel() {
        let mut booster = create_mock_booster();
        booster.cfg.objective = Objective::LogLoss;
        booster.base_score = 0.0;
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let proba = booster.predict_proba_columnar(&data, true, false);
        assert_eq!(proba.len(), 2);
    }

    #[test]
    fn test_predict_nodes_columnar() {
        let booster = create_mock_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let nodes = booster.predict_nodes_columnar(&data, false);
        assert_eq!(nodes.len(), 1); // 1 tree
        assert_eq!(nodes[0].len(), 2); // 2 rows
    }

    #[test]
    fn test_predict_contributions_average_parallel() {
        let booster = create_mock_booster();
        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, true);
        assert_eq!(contribs.len(), 6); // 2 rows * (2 features + 1 bias)
    }

    #[test]
    fn test_predict_contributions_weight_parallel() {
        let booster = create_mock_booster();
        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Weight, true);
        assert_eq!(contribs.len(), 6);
    }

    #[test]
    fn test_predict_contributions_columnar_average() {
        let booster = create_mock_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let contribs = booster.predict_contributions_columnar(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), 6);
    }

    #[test]
    fn test_predict_contributions_columnar_average_parallel() {
        let booster = create_mock_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let contribs = booster.predict_contributions_columnar(&data, ContributionsMethod::Average, true);
        assert_eq!(contribs.len(), 6);
    }

    #[test]
    fn test_predict_contributions_columnar_all_methods() {
        let booster = create_mock_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let methods = vec![
            ContributionsMethod::Weight,
            ContributionsMethod::BranchDifference,
            ContributionsMethod::MidpointDifference,
            ContributionsMethod::ModeDifference,
            ContributionsMethod::Shapley,
        ];
        for method in methods {
            let contribs = booster.predict_contributions_columnar(&data, method, false);
            assert_eq!(contribs.len(), 6);
        }
    }

    #[test]
    fn test_predict_contributions_columnar_probability_change() {
        let mut booster = create_mock_booster();
        booster.cfg.objective = Objective::LogLoss;
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let contribs = booster.predict_contributions_columnar(&data, ContributionsMethod::ProbabilityChange, false);
        assert_eq!(contribs.len(), 6);
    }

    #[test]
    fn test_predict_contributions_columnar_probability_change_parallel() {
        let mut booster = create_mock_booster();
        booster.cfg.objective = Objective::LogLoss;
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let contribs = booster.predict_contributions_columnar(&data, ContributionsMethod::ProbabilityChange, true);
        assert_eq!(contribs.len(), 6);
    }

    #[test]
    fn test_predict_contributions_tree_alone_columnar_parallel() {
        let booster = create_mock_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let contribs = booster.predict_contributions_columnar(&data, ContributionsMethod::Weight, true);
        assert_eq!(contribs.len(), 6);
    }

    #[test]
    fn test_multi_output_predict_columnar() {
        let booster = create_mock_multi_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let preds = booster.predict_columnar(&data, false);
        assert_eq!(preds.len(), 4); // 2 rows * 2 boosters
    }

    #[test]
    fn test_multi_output_predict_proba_columnar() {
        let booster = create_mock_multi_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let proba = booster.predict_proba_columnar(&data, false);
        assert_eq!(proba.len(), 4);
        // Probabilities should sum to 1 per row
    }

    #[test]
    fn test_multi_output_predict_nodes() {
        let booster = create_mock_multi_booster();
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let nodes = booster.predict_nodes(&data, false);
        assert_eq!(nodes.len(), 2); // 2 boosters
    }

    #[test]
    fn test_multi_output_predict_nodes_columnar() {
        let booster = create_mock_multi_booster();
        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let nodes = booster.predict_nodes_columnar(&data, false);
        assert_eq!(nodes.len(), 2); // 2 boosters
    }

    #[test]
    fn test_multi_output_predict_contributions() {
        let booster = create_mock_multi_booster();
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Weight, false);
        assert_eq!(contribs.len(), 6); // 1 row * 3 items * 2 boosters
    }

    #[test]
    fn test_multi_output_predict_intervals() {
        let booster = create_mock_multi_booster();
        let data = Matrix::new(&[1.0, 2.0], 1, 2);
        let intervals = booster.predict_intervals(&data, false);
        // No calibration params set, so intervals should be empty
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_predict_contributions_probability_change_parallel() {
        let mut booster = create_mock_booster();
        booster.cfg.objective = Objective::LogLoss;
        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::ProbabilityChange, true);
        assert_eq!(contribs.len(), 6);
    }
}
