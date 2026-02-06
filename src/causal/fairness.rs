use crate::metrics::evaluation::Metric;
use crate::objective_functions::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

/// Fairness Objective (In-processing)
///
/// Adds a regularization term to the loss to penalize predictions that correlate with a sensitive attribute S,
/// conditional on the target Y (Equal Odds) or marginally (Demographic Parity).
///
/// Loss = LogLoss + \lambda * Correlation(Yhat, S)
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessObjective {
    pub sensitive_attr: Vec<i32>, // Group membership
    pub lambda: f32,              // Penalty strength
}

impl FairnessObjective {
    pub fn new(sensitive_attr: Vec<i32>, lambda: f32) -> Self {
        Self { sensitive_attr, lambda }
    }
}

impl ObjectiveFunction for FairnessObjective {
    fn loss(&self, y: &[f64], yhat: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        // Simplified: Return Standard LogLoss
        // We only modify gradients for training.
        y.iter()
            .zip(yhat.iter())
            .map(|(y_i, yhat_i)| {
                let p = 1.0 / (1.0 + (-yhat_i).exp());
                // - (y log p + (1-y) log (1-p)) since we can't easily visualize the fairness penalty here per sample
                let score = -(y_i * p.ln() + (1.0 - y_i) * (1.0 - p).ln());
                score as f32
            })
            .collect()
    }

    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        _sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        // L = LogLoss + \lambda * (Mean(p | S=1) - Mean(p | S=0))^2  (Demographic Parity Proxy)
        // dL/dp = (p - y) + 2 * \lambda * (Mean_diff) * (1/N_s * Indicator)

        let n = y.len();
        let mut grad = Vec::with_capacity(n);
        let mut hess = Vec::with_capacity(n);

        // Calculate current means per group
        let mut s1_sum: f64 = 0.0;
        let mut s1_count: f64 = 0.0;
        let mut s0_sum: f64 = 0.0;
        let mut s0_count: f64 = 0.0;

        let mut probs = Vec::with_capacity(n);

        for (i, &yh) in yhat.iter().enumerate().take(n) {
            let p = 1.0 / (1.0 + (-yh).exp());
            probs.push(p);
            if self.sensitive_attr[i] == 1 {
                s1_sum += p;
                s1_count += 1.0;
            } else {
                s0_sum += p;
                s0_count += 1.0;
            }
        }

        // Use partial_max or manual check for f64
        let div_s1 = if s1_count < 1.0 { 1.0 } else { s1_count };
        let div_s0 = if s0_count < 1.0 { 1.0 } else { s0_count };

        let mean_s1 = s1_sum / div_s1;
        let mean_s0 = s0_sum / div_s0;
        let diff = mean_s1 - mean_s0; // Parity difference

        for i in 0..n {
            let p = probs[i];
            let y_i = y[i];

            // Standard LogLoss derivative w.r.t logits
            let mut g = p - y_i;

            // Fairness penalty derivative
            // d(Diff^2)/dscore = 2 * Diff * dDiff/dscore
            // dDiff/dscore_i:
            //   If i in S1: d(1/N1 * sum p)/dscore = 1/N1 * p(1-p)
            //   If i in S0: d(-1/N0 * sum p)/dscore = -1/N0 * p(1-p)

            let fairness_grad = if self.sensitive_attr[i] == 1 {
                2.0 * (self.lambda as f64) * diff * (1.0 / div_s1) * p * (1.0 - p)
            } else {
                2.0 * (self.lambda as f64) * diff * (-1.0 / div_s0) * p * (1.0 - p)
            };

            g += fairness_grad;

            let h = p * (1.0 - p); // Standard Hessian approx

            grad.push(g as f32);
            hess.push(h as f32);
        }

        (grad, Some(hess))
    }

    fn initial_value(&self, _y: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        0.0
    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss
    }
}
