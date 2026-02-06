//! R-Learner Objective
//!
//! Custom objective function for the effect model in the R-Learner
//! meta-algorithm. Minimizes the R-Loss for CATE estimation.
use crate::metrics::evaluation::Metric;
use crate::objective_functions::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

/// Minimum propensity clip to avoid division-by-zero or extreme weights.
const PROPENSITY_CLIP_MIN: f64 = 1e-6;
/// Maximum propensity clip.
const PROPENSITY_CLIP_MAX: f64 = 1.0 - 1e-6;
/// Minimum hessian floor to stabilize the R-Learner when propensity
/// residuals are very small (overlap violations).
const HESSIAN_FLOOR: f64 = 1e-6;

/// R-Learner objective for estimating the Conditional Average Treatment Effect (CATE).
///
/// Minimizes the R-Loss:
/// $L = \sum \bigl((Y - \hat{\mu}(X)) - \tau(X)(W - \hat{p}(X))\bigr)^2$
///
/// where $\tau(X)$ is the treatment effect being learned.
///
/// Propensity scores are automatically clipped to
/// `[PROPENSITY_CLIP_MIN, PROPENSITY_CLIP_MAX]` and the hessian is floored
/// at `HESSIAN_FLOOR` to prevent numerical blow-up in regions of poor overlap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLearnerObjective {
    /// Observed treatment assignments.
    pub treatment: Vec<f64>,
    /// Predicted outcome $\hat{\mu}(x)$ from the nuisance outcome model.
    pub outcome_predicted: Vec<f64>,
    /// Predicted treatment probability $\hat{p}(x)$ from the nuisance propensity model.
    pub treatment_predicted: Vec<f64>,
}

impl RLearnerObjective {
    /// Create a new `RLearnerObjective`.
    ///
    /// # Panics
    /// Panics if the three input vectors have different lengths.
    pub fn new(treatment: Vec<f64>, outcome_predicted: Vec<f64>, treatment_predicted: Vec<f64>) -> Self {
        assert_eq!(treatment.len(), outcome_predicted.len());
        assert_eq!(treatment.len(), treatment_predicted.len());
        Self {
            treatment,
            outcome_predicted,
            treatment_predicted,
        }
    }
}

impl ObjectiveFunction for RLearnerObjective {
    fn loss(&self, y: &[f64], yhat: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        // L_i = ( (y_i - mu_i) - tau_i * (w_i - p_i) )^2
        y.iter()
            .zip(yhat.iter())
            .zip(self.treatment.iter())
            .zip(self.outcome_predicted.iter())
            .zip(self.treatment_predicted.iter())
            .map(|((((y_i, tau_i), w_i), mu_i), p_i)| {
                let y_res = y_i - mu_i;
                let p_clipped = p_i.clamp(PROPENSITY_CLIP_MIN, PROPENSITY_CLIP_MAX);
                let w_res = w_i - p_clipped;
                let diff = y_res - tau_i * w_res;
                (diff * diff) as f32
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
        // L = ( (y - \mu) - \tau * (w - p) )^2
        // dL/dtau = -2 * (w - p) * ( (y - \mu) - \tau * (w - p) )
        // d^2L/dtau^2 = 2 * (w - p)^2

        let n = y.len();
        let mut grad = Vec::with_capacity(n);
        let mut hess = Vec::with_capacity(n);

        for i in 0..n {
            let y_res = y[i] - self.outcome_predicted[i];
            let p_clipped = self.treatment_predicted[i].clamp(PROPENSITY_CLIP_MIN, PROPENSITY_CLIP_MAX);
            let w_res = self.treatment[i] - p_clipped;
            let tau = yhat[i];

            // Derivative of 0.5 * ((y-mu) - tau*(w-p))^2 w.r.t. tau:
            //   g = -(w-p) * ((y-mu) - tau*(w-p))
            //     = -w_res * y_res + tau * w_res^2
            //   h = (w-p)^2
            let g = -w_res * y_res + tau * w_res * w_res;
            let h = (w_res * w_res).max(HESSIAN_FLOOR);

            grad.push(g as f32);
            hess.push(h as f32);
        }

        (grad, Some(hess))
    }

    fn initial_value(&self, _y: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        0.0 // Start with 0 treatment effect
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }
}
