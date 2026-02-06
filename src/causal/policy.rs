use crate::metrics::evaluation::Metric;
use crate::objective_functions::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

/// Policy Gradient Objective (Inverse Propensity Weighting)
///
/// Optimizes a policy score $F(x)$ to maximize the expected reward.
/// We use the transformation:
/// Minimize $L = - \frac{Y \cdot \mathbb{I}(W = \text{sign}(F(x)))}{P(W|\text{policy})}$
///
/// For binary treatment W \in {0, 1}:
/// If W=1 observed: We want F(x) > 0 (high). Reward contribution Y / P(W=1|X). Gradient pushes F(x) up.
/// If W=0 observed: We want F(x) < 0 (low). Reward contribution Y / P(W=0|X). Gradient pushes F(x) down.
///
/// Simplified Loss (Policy Gradient):
/// L = - (Y / p) * log(\sigma(sign(W) * F(x)))
/// But standard GBM regression minimizes L2.
///
/// We adopt the **Athey & Wager (2021) Policy Learning** approach or generic Gradient/Hessian specification.
///
/// Here, we implement a custom objective that treats the problem as Weighted Classification
/// where the weight is |Y| / P(W|X) and the label is sign(Y) * sign(W - 0.5) (simplified).
///
/// Actually, a robust way is "Learning to Search" or simply:
/// Gradient = - Reward * (1 - p) if Action=1 was taken.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyObjective {
    pub treatment: Vec<u8>,   // 0 or 1
    pub propensity: Vec<f64>, // P(W=1|X)
}

impl PolicyObjective {
    pub fn new(treatment: Vec<u8>, propensity: Vec<f64>) -> Self {
        Self { treatment, propensity }
    }
}

impl ObjectiveFunction for PolicyObjective {
    fn loss(&self, y: &[f64], yhat: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        // Proxy loss for visualization (negative reward)
        // This is not exactly what's minimized but serves as a metric.
        y.iter()
            .zip(yhat.iter())
            .map(|(reward, score)| {
                // simple logistic policy probability
                let pi = 1.0 / (1.0 + (-score).exp());
                -(*reward) as f32 * pi as f32
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
        let n = y.len();
        let mut grad = Vec::with_capacity(n);
        let mut hess = Vec::with_capacity(n);

        for i in 0..n {
            let w = self.treatment[i];
            let p = self.propensity[i].clamp(1e-3, 1.0 - 1e-3);
            let reward = y[i];

            // Current policy probability P(Action=1) = sigma(yhat)
            let score = yhat[i];
            let sigma = 1.0 / (1.0 + (-score).exp());

            // IPW Gradient
            // If W=1: We observed reward Y. We want to increase sigma.
            // dReward/dScore = Y/p * sigma * (1-sigma)
            // We want to MAXIMIZE reward, so minimize negative reward.
            // Grad = - (Y/p) * sigma * (1-sigma)

            // For stability, we typically use the "Doubly Robust" or just IPW score as a weight
            // in a weighted classification problem aimed at the "optimal" action.

            // Implementation:
            // g = sigma - 1 (if optimal action is 1),
            //     sigma - 0 (if optimal action is 0).
            // weighted by |Y|/p ?

            // Let's use the explicit Policy Gradient form:
            // G = - (Y * (W - p)) ... wait, that's diff.

            // Simple approach:
            // Transform to classification.
            // If Y > 0 and W=1 -> Label=1. Weight = Y/p.
            // If Y > 0 and W=0 -> Label=0. Weight = Y/(1-p).
            // If Y < 0 and W=1 -> Label=0. Weight = -Y/p. (Avoid Action 1)
            // If Y < 0 and W=0 -> Label=1. Weight = -Y/(1-p). (Avoid Action 0 -> Do Action 1)

            let (target, weight) = if w == 1 {
                if reward >= 0.0 {
                    (1.0, reward / p)
                } else {
                    (0.0, -reward / p)
                }
            } else {
                if reward >= 0.0 {
                    (0.0, reward / (1.0 - p))
                } else {
                    (1.0, -reward / (1.0 - p))
                }
            };

            // LogLoss Gradient w.r.t score (logits): p - y
            // Weighted: weight * (sigma - target)
            let g = weight * (sigma - target);
            let h = weight * sigma * (1.0 - sigma);

            grad.push(g as f32);
            hess.push(h as f32); // Constant Hessian for stability? Or actual?
        }

        (grad, Some(hess))
    }

    fn initial_value(&self, _y: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        0.0
    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss // It's treated as a classification problem
    }
}
