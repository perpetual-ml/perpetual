//! Fairness Objective
//!
//! An in-processing fairness objective that adds a regularization penalty
//! to the standard log-loss gradient.  Two modes are supported:
//!
//! * **Demographic Parity** — penalizes correlation between $\hat{Y}$ and $S$.
//! * **Equalized Odds** — penalizes correlation between $\hat{Y}$ and $S$
//!   *conditionally* within each class of the true label $Y$.
use crate::metrics::evaluation::Metric;
use crate::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

/// Which fairness criterion to enforce.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FairnessType {
    /// Penalize overall disparity: $\text{Mean}(\hat{p}|S\!=\!1) - \text{Mean}(\hat{p}|S\!=\!0)$.
    DemographicParity,
    /// Penalize disparity within each label class (Equalized Odds):
    /// $\sum_{y \in \{0,1\}} \bigl(\text{Mean}(\hat{p}|S\!=\!1, Y\!=\!y) - \text{Mean}(\hat{p}|S\!=\!0, Y\!=\!y)\bigr)^2$.
    EqualizedOdds,
}

/// Fairness Objective (In-processing).
///
/// Adds a regularization term to log-loss to penalize predictions that
/// correlate with a sensitive attribute $S$.
///
/// $$L = \text{LogLoss} + \lambda \cdot \text{Penalty}(\hat{Y}, S)$$
///
/// The penalty form depends on the chosen [`FairnessType`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessObjective {
    /// Group membership vector (e.g., 0 or 1 for each sample).
    pub sensitive_attr: Vec<i32>,
    /// Penalty strength for the fairness regularization term.
    pub lambda: f32,
    /// Fairness criterion to enforce.
    pub fairness_type: FairnessType,
}

impl FairnessObjective {
    /// Create a new `FairnessObjective` with Demographic Parity.
    ///
    /// * `sensitive_attr` - Binary group membership for each sample.
    /// * `lambda` - Strength of the fairness penalty.
    pub fn new(sensitive_attr: Vec<i32>, lambda: f32) -> Self {
        Self {
            sensitive_attr,
            lambda,
            fairness_type: FairnessType::DemographicParity,
        }
    }

    /// Create a new `FairnessObjective` with a specific fairness type.
    pub fn with_type(sensitive_attr: Vec<i32>, lambda: f32, fairness_type: FairnessType) -> Self {
        Self {
            sensitive_attr,
            lambda,
            fairness_type,
        }
    }
}

/// Accumulator for group-conditional statistics.
#[derive(Default)]
struct GroupStats {
    sum: f64,
    count: f64,
}

impl GroupStats {
    fn mean(&self) -> f64 {
        if self.count < 1.0 { 0.0 } else { self.sum / self.count }
    }
    fn safe_count(&self) -> f64 {
        if self.count < 1.0 { 1.0 } else { self.count }
    }
}

impl ObjectiveFunction for FairnessObjective {
    fn loss(&self, y: &[f64], yhat: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        let n = y.len();
        let probs: Vec<f64> = yhat.iter().map(|yh| 1.0 / (1.0 + (-yh).exp())).collect();
        let mut loss = Vec::with_capacity(n);

        // Standard LogLoss
        for i in 0..n {
            let y_i = y[i];
            let p = probs[i];
            let score = -(y_i * p.max(1e-15).ln() + (1.0 - y_i) * (1.0 - p).max(1e-15).ln());
            loss.push(score as f32);
        }

        let n_f64 = n as f64;

        match &self.fairness_type {
            FairnessType::DemographicParity => {
                let mut s1 = GroupStats::default();
                let mut s0 = GroupStats::default();
                for (i, &p) in probs.iter().enumerate() {
                    if self.sensitive_attr[i] == 1 {
                        s1.sum += p;
                        s1.count += 1.0;
                    } else {
                        s0.sum += p;
                        s0.count += 1.0;
                    }
                }
                let diff = s1.mean() - s0.mean();
                let penalty = (self.lambda as f64) * diff * diff * n_f64;

                // We evenly distribute the squared diff penalty amongst the batch
                // essentially returning sum(logloss) + lambda * diff^2 * N when globally reduced.
                // In Perpetual, the loss output vector is summed. So each element is + penalty/N
                let p_per_n = (penalty / n_f64) as f32;
                for l in loss.iter_mut() {
                    *l += p_per_n;
                }
            }
            FairnessType::EqualizedOdds => {
                let mut s0_y0 = GroupStats::default();
                let mut s0_y1 = GroupStats::default();
                let mut s1_y0 = GroupStats::default();
                let mut s1_y1 = GroupStats::default();

                for i in 0..n {
                    let p = probs[i];
                    let label = if y[i] >= 0.5 { 1 } else { 0 };
                    let group = self.sensitive_attr[i];
                    match (group, label) {
                        (1, 1) => {
                            s1_y1.sum += p;
                            s1_y1.count += 1.0;
                        }
                        (1, 0) => {
                            s1_y0.sum += p;
                            s1_y0.count += 1.0;
                        }
                        (_, 1) => {
                            s0_y1.sum += p;
                            s0_y1.count += 1.0;
                        }
                        (_, 0) => {
                            s0_y0.sum += p;
                            s0_y0.count += 1.0;
                        }
                        _ => unreachable!(),
                    }
                }

                let diff_y0 = s1_y0.mean() - s0_y0.mean();
                let diff_y1 = s1_y1.mean() - s0_y1.mean();

                let penalty = (self.lambda as f64) * (diff_y0 * diff_y0 + diff_y1 * diff_y1) * n_f64;
                let p_per_n = (penalty / n_f64) as f32;

                for l in loss.iter_mut() {
                    *l += p_per_n;
                }
            }
        }

        loss
    }

    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        _sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let n = y.len();
        let mut grad = Vec::with_capacity(n);
        let mut hess = Vec::with_capacity(n);

        // Compute predicted probabilities.
        let probs: Vec<f64> = yhat.iter().map(|yh| 1.0 / (1.0 + (-yh).exp())).collect();

        // Scale fairness gradient by N to match LogLoss scale (sum over N).
        let n_f64 = n as f64;

        match &self.fairness_type {
            FairnessType::DemographicParity => {
                // Compute mean predicted probability per sensitive group.
                let mut s1 = GroupStats::default();
                let mut s0 = GroupStats::default();
                for (i, &p) in probs.iter().enumerate() {
                    if self.sensitive_attr[i] == 1 {
                        s1.sum += p;
                        s1.count += 1.0;
                    } else {
                        s0.sum += p;
                        s0.count += 1.0;
                    }
                }
                let diff = s1.mean() - s0.mean();

                for i in 0..n {
                    let p = probs[i];
                    let dp = p * (1.0 - p); // dsigma/dscore
                    let mut g = p - y[i]; // standard LogLoss gradient

                    // d(diff^2)/dscore_i
                    let fairness_grad = if self.sensitive_attr[i] == 1 {
                        2.0 * (self.lambda as f64) * diff * (1.0 / s1.safe_count()) * dp * n_f64
                    } else {
                        2.0 * (self.lambda as f64) * diff * (-1.0 / s0.safe_count()) * dp * n_f64
                    };
                    g += fairness_grad;

                    grad.push(g as f32);
                    hess.push(dp as f32);
                }
            }
            FairnessType::EqualizedOdds => {
                // Compute conditional mean per (sensitive group, true label) cell.
                // Cells: (S=0,Y=0), (S=0,Y=1), (S=1,Y=0), (S=1,Y=1)
                let mut s0_y0 = GroupStats::default();
                let mut s0_y1 = GroupStats::default();
                let mut s1_y0 = GroupStats::default();
                let mut s1_y1 = GroupStats::default();

                for i in 0..n {
                    let p = probs[i];
                    let label = if y[i] >= 0.5 { 1 } else { 0 };
                    let group = self.sensitive_attr[i];
                    match (group, label) {
                        (1, 1) => {
                            s1_y1.sum += p;
                            s1_y1.count += 1.0;
                        }
                        (1, 0) => {
                            s1_y0.sum += p;
                            s1_y0.count += 1.0;
                        }
                        (_, 1) => {
                            s0_y1.sum += p;
                            s0_y1.count += 1.0;
                        }
                        (_, 0) => {
                            s0_y0.sum += p;
                            s0_y0.count += 1.0;
                        }
                        _ => unreachable!(),
                    }
                }

                let diff_y0 = s1_y0.mean() - s0_y0.mean();
                let diff_y1 = s1_y1.mean() - s0_y1.mean();

                for i in 0..n {
                    let p = probs[i];
                    let dp = p * (1.0 - p);
                    let mut g = p - y[i]; // standard LogLoss gradient

                    let label = if y[i] >= 0.5 { 1 } else { 0 };
                    let (diff, cnt_s1, cnt_s0) = if label == 1 {
                        (diff_y1, s1_y1.safe_count(), s0_y1.safe_count())
                    } else {
                        (diff_y0, s1_y0.safe_count(), s0_y0.safe_count())
                    };

                    let fairness_grad = if self.sensitive_attr[i] == 1 {
                        2.0 * (self.lambda as f64) * diff * (1.0 / cnt_s1) * dp * n_f64
                    } else {
                        2.0 * (self.lambda as f64) * diff * (-1.0 / cnt_s0) * dp * n_f64
                    };
                    g += fairness_grad;

                    grad.push(g as f32);
                    hess.push(dp as f32);
                }
            }
        }

        (grad, Some(hess))
    }

    fn initial_value(&self, y: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let p = mean.clamp(1e-15, 1.0 - 1e-15);
        (p / (1.0 - p)).ln()
    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss
    }
}
