//! Predictions
//!
//!

use crate::booster::config::ContributionsMethod;
use crate::objective_functions::objective::Objective;
use crate::MultiOutputBooster;
use crate::{
    decision_tree::tree::Tree, shapley::predict_contributions_row_shapley, utils::odds, Matrix, PerpetualBooster,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

impl PerpetualBooster {
    /// Generate predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let mut init_preds = vec![self.base_score; data.rows];
        self.get_prediction_trees().iter().for_each(|tree| {
            for (p_, val) in init_preds
                .iter_mut()
                .zip(tree.predict(data, parallel, &self.cfg.missing))
            {
                *p_ += val;
            }
        });
        init_preds
    }

    /// Generate probabilities on data using the gradient booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict_proba(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let preds = self.predict(data, parallel);
        if parallel {
            preds.par_iter().map(|p| 1.0 / (1.0 + (-p).exp())).collect()
        } else {
            preds.iter().map(|p| 1.0 / (1.0 + (-p).exp())).collect()
        }
    }

    /// Predict the contributions matrix for the provided dataset.
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

    /// Generate probabilities on data using the multi-output booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict_proba(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let log_odds = self.predict(data, parallel);
        let data_log_odds = Matrix::new(&log_odds, data.rows, data.cols);
        let mut preds = Vec::with_capacity(log_odds.len());
        for row in 0..data.rows {
            let y_p_exp = data_log_odds.get_row(row).iter().map(|e| e.exp()).collect::<Vec<f64>>();
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
}
