use crate::metrics::evaluation::Metric;
use crate::objective_functions::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLearnerObjective {
    pub treatment: Vec<f64>,
    pub outcome_predicted: Vec<f64>,   // \hat{\mu}(x)
    pub treatment_predicted: Vec<f64>, // \hat{p}(x)
}

impl RLearnerObjective {
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
        // L = \sum ( (y - \mu) - \tau(x) * (w - p) )^2
        // y here is the actual outcome.
        // yhat is \tau(x).

        y.iter()
            .zip(yhat.iter())
            .zip(self.treatment.iter())
            .zip(self.outcome_predicted.iter())
            .zip(self.treatment_predicted.iter())
            .map(|((((y_i, tau_i), w_i), mu_i), p_i)| {
                let y_res = y_i - mu_i;
                let w_res = w_i - p_i;
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
            let w_res = self.treatment[i] - self.treatment_predicted[i];
            let tau = yhat[i];

            // Gradient: -2 * w_res * (y_res - tau * w_res)
            // But usually we drop the factor of 2 for optimization or include 0.5 in loss
            // SquaredLoss in core uses: g = yhat - y, h = 1 (for 0.5 * (y-yhat)^2)
            // Let's match that scale.
            // Metric to minimize: 0.5 * (y_target - y_pred)^2
            // Our residual target is y_res / w_res (if w_res != 0)
            // weighted by w_res^2.

            // Let's compute exact derivative of 0.5 * ((y-mu) - tau*(w-p))^2
            // d/dtau = -(w-p) * (y_res - tau*w_res)
            //        = -w_res*y_res + tau*w_res^2

            let g = -w_res * y_res + tau * w_res * w_res;
            let h = w_res * w_res;

            // Stabilize Hessian?
            // If w_res is very small (propensity near 0 or 1), h is small.
            // This effectively ignores those samples, which is correct for R-learner.

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
