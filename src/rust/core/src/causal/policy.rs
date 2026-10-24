//! Policy Gradient Objective
//!
//! An inverse-propensity-weighted (IPW) objective that learns a policy
//! maximizing expected reward via gradient boosting.
//!
//! Two modes are supported:
//!
//! * **IPW** – Standard Inverse Propensity Weighting.
//! * **AIPW** (Augmented IPW / Doubly Robust) – Reduces variance by
//!   incorporating a baseline outcome model $\hat{\mu}(X)$.
use crate::metrics::evaluation::Metric;
use crate::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

/// Minimum propensity clip for numerical safety.
const PROPENSITY_CLIP: f64 = 1e-3;

/// Policy learning mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyMode {
    /// Standard Inverse Propensity Weighting.
    IPW,
    /// Augmented / Doubly Robust IPW with baseline outcome predictions.
    AIPW {
        /// Predicted baseline outcome $\hat{\mu}_1(X)$ for treated.
        mu_hat_1: Vec<f64>,
        /// Predicted baseline outcome $\hat{\mu}_0(X)$ for control.
        mu_hat_0: Vec<f64>,
    },
}

/// Policy Gradient Objective (Inverse Propensity Weighting).
///
/// Optimizes a policy score $F(x)$ to maximize the expected reward using
/// the Athey & Wager (2021) Policy Learning approach.
///
/// The learned policy assigns treatment $W = 1$ when $\sigma(F(x)) > 0.5$
/// (equivalently $F(x) > 0$).
///
/// For binary treatment $W \in \{0, 1\}$:
/// - The IPW pseudo-outcome for each sample is:
///   $$\Gamma_i = \frac{Y_i \cdot W_i}{p_i} - \frac{Y_i \cdot (1 - W_i)}{1 - p_i}$$
/// - The objective reduces to logistic regression weighted by $|\Gamma_i|$.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyObjective {
    /// Observed binary treatment assignment (0 or 1).
    pub treatment: Vec<u8>,
    /// Estimated propensity score $P(W=1|X)$ for each sample.
    pub propensity: Vec<f64>,
    /// Policy mode: IPW or AIPW.
    pub mode: PolicyMode,
}

impl PolicyObjective {
    /// Create a new IPW `PolicyObjective`.
    ///
    /// * `treatment` - Binary treatment assignments.
    /// * `propensity` - Estimated $P(W=1|X)$ for each sample.
    pub fn new(treatment: Vec<u8>, propensity: Vec<f64>) -> Self {
        Self {
            treatment,
            propensity,
            mode: PolicyMode::IPW,
        }
    }

    /// Create a new AIPW (Doubly Robust) `PolicyObjective`.
    ///
    /// * `treatment` - Binary treatment assignments.
    /// * `propensity` - Estimated $P(W=1|X)$ for each sample.
    /// * `mu_hat_1` - Predicted baseline outcome $\hat{\mu}_1(X)$ for treated.
    /// * `mu_hat_0` - Predicted baseline outcome $\hat{\mu}_0(X)$ for control.
    pub fn new_aipw(treatment: Vec<u8>, propensity: Vec<f64>, mu_hat_1: Vec<f64>, mu_hat_0: Vec<f64>) -> Self {
        Self {
            treatment,
            propensity,
            mode: PolicyMode::AIPW { mu_hat_1, mu_hat_0 },
        }
    }

    /// Compute the IPW or AIPW pseudo-outcome $\Gamma_i$ for each sample.
    ///
    /// $$\Gamma_i^{IPW} = \frac{Y_i W_i}{p_i} - \frac{Y_i (1-W_i)}{1-p_i}$$
    ///
    /// $$\Gamma_i^{AIPW} = (\hat{\mu}_1 - \hat{\mu}_0)
    ///     + \frac{W_i (Y_i - \hat{\mu}_1)}{p_i}
    ///     - \frac{(1-W_i)(Y_i - \hat{\mu}_0)}{1-p_i}$$
    fn pseudo_outcome(&self, y: &[f64]) -> Vec<f64> {
        let n = y.len();
        let mut gamma = Vec::with_capacity(n);

        for i in 0..n {
            let w = self.treatment[i] as f64;
            let p = self.propensity[i].clamp(PROPENSITY_CLIP, 1.0 - PROPENSITY_CLIP);
            let yi = y[i];

            let g = match &self.mode {
                PolicyMode::IPW => {
                    // Gamma = Y*W/p - Y*(1-W)/(1-p)
                    yi * w / p - yi * (1.0 - w) / (1.0 - p)
                }
                PolicyMode::AIPW { mu_hat_1, mu_hat_0 } => {
                    let m1 = mu_hat_1[i];
                    let m0 = mu_hat_0[i];

                    // AIPW score = (m1 - m0) + W(Y-m1)/p - (1-W)(Y-m0)/(1-p)
                    let term1 = m1 - m0;
                    let term2 = w * (yi - m1) / p;
                    let term3 = (1.0 - w) * (yi - m0) / (1.0 - p);

                    term1 + term2 - term3
                }
            };
            gamma.push(g);
        }
        gamma
    }
}

impl ObjectiveFunction for PolicyObjective {
    fn loss(&self, y: &[f64], yhat: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        // Weighted logistic loss: -|Gamma| * [ target * log(sigma) + (1-target) * log(1-sigma) ]
        let gamma = self.pseudo_outcome(y);
        gamma
            .iter()
            .zip(yhat.iter())
            .map(|(g, score)| {
                let sigma = 1.0 / (1.0 + (-score).exp());
                let target = if *g >= 0.0 { 1.0 } else { 0.0 };
                let w = g.abs();
                let loss = -(target * sigma.max(1e-15).ln() + (1.0 - target) * (1.0 - sigma).max(1e-15).ln());
                (w * loss) as f32
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
        // Transform to weighted binary classification via pseudo-outcomes.
        //
        // Gamma_i > 0  →  target = 1, weight = |Gamma_i|  (treat is beneficial)
        // Gamma_i < 0  →  target = 0, weight = |Gamma_i|  (treat is harmful)
        //
        // Gradient:  w_i * (sigma(F) - target_i)
        // Hessian:   w_i * sigma(F) * (1 - sigma(F))
        let n = y.len();
        let gamma = self.pseudo_outcome(y);
        let mut grad = Vec::with_capacity(n);
        let mut hess = Vec::with_capacity(n);

        for i in 0..n {
            let score = yhat[i];
            let sigma = 1.0 / (1.0 + (-score).exp());

            let target = if gamma[i] >= 0.0 { 1.0 } else { 0.0 };
            let weight = gamma[i].abs();

            let g = weight * (sigma - target);
            let h = weight * sigma * (1.0 - sigma);

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
