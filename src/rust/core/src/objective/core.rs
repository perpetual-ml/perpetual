use crate::{
    metrics::evaluation::Metric,
    objective::{
        AbsoluteLoss, AdaptiveHuberLoss, BrierLoss, CrossEntropyLambdaLoss, CrossEntropyLoss, FairLoss, GammaLoss,
        HingeLoss, HuberLoss, ListNetLoss, LogLoss, MapeLoss, PoissonLoss, QuantileLoss, SquaredLogLoss, SquaredLoss,
        TweedieLoss,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Trait defining a custom objective function.
///
/// Implement this trait to define your own loss function and gradient calculation.
///
/// Only [`loss`](ObjectiveFunction::loss) and [`gradient`](ObjectiveFunction::gradient)
/// are required — the remaining methods have sensible defaults:
///
/// * `initial_value` — weighted mean of `y` (suitable for most regression objectives).
/// * `default_metric` — `Metric::RootMeanSquaredError`.
/// * `gradient_and_loss` — calls `gradient()` then `loss()` (override for fused implementations).
pub trait ObjectiveFunction: Send + Sync {
    /// Per-sample loss.
    ///
    /// # Arguments
    /// * `y` – true target values.
    /// * `yhat` – predicted values (log-odds or raw scores).
    /// * `sample_weight` – optional per-sample weights.
    /// * `group` – optional group sizes (for ranking).
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> Vec<f32>;

    /// Per-sample gradient and (optional) hessian.
    ///
    /// Returns `(gradient, hessian)`.
    /// * `gradient` – first derivative of loss w.r.t. prediction.
    /// * `hessian` – second derivative. Return `None` when the hessian is constant
    ///   (e.g. 1.0 for squared loss) to enable an optimized code path.
    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>);

    /// Initial prediction (base score) before any trees are added.
    ///
    /// Default: weighted mean of `y`.
    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        match sample_weight {
            Some(w) => {
                let sw: f64 = w.iter().sum();
                y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / sw
            }
            None => y.iter().sum::<f64>() / y.len() as f64,
        }
    }

    /// Default evaluation metric for this objective.
    ///
    /// Default: `Metric::RootMeanSquaredError`.
    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }

    /// Gradient, hessian **and** loss in a single pass over the data.
    ///
    /// The default implementation calls [`gradient`](ObjectiveFunction::gradient)
    /// and [`loss`](ObjectiveFunction::loss) separately. Override when the two
    /// share intermediate computations (e.g. the sigmoid in log-loss).
    fn gradient_and_loss(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>, Vec<f32>) {
        let (g, h) = self.gradient(y, yhat, sample_weight, group);
        let l = self.loss(y, yhat, sample_weight, group);
        (g, h, l)
    }

    /// In-place version of [`gradient_and_loss`](ObjectiveFunction::gradient_and_loss).
    ///
    /// Writes into pre-allocated buffers to avoid per-iteration allocations.
    /// Default implementation delegates to `gradient_and_loss` with fresh allocs.
    #[allow(clippy::too_many_arguments)]
    fn gradient_and_loss_into(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        grad: &mut [f32],
        hess: &mut Option<Vec<f32>>,
        loss: &mut [f32],
    ) {
        let (g, h, l) = self.gradient_and_loss(y, yhat, sample_weight, group);
        grad.copy_from_slice(&g);
        *hess = h;
        loss.copy_from_slice(&l);
    }

    /// Whether this objective requires batch evaluation (e.g. for Python custom objectives).
    ///
    /// If true, `loss()` will be called with the full vector instead of `loss_single()` in a loop.
    /// Defaults to `true` as `loss_single` is an optional optimization.
    fn requires_batch_evaluation(&self) -> bool {
        true
    }
}

/// The Objective function to minimize during training.
///
/// Each objective corresponds to a specific loss function and gradient calculation.
/// Choose the objective that best matches your problem type.
#[derive(Serialize, Deserialize, Clone)]
pub enum Objective {
    /// LogLoss for binary classification.
    /// Minimizes `- (y * log(p) + (1-y) * log(1-p))`.
    /// Target `y` should be 0 or 1.
    LogLoss,
    /// Brier Score for probabilistic binary classification.
    /// Minimizes `(y - p)^2` where `p = sigmoid(yhat)`.
    BrierLoss,
    /// Squared Error Loss for regression.
    /// Minimizes `0.5 * (y - yhat)^2`.
    /// The derivative is `yhat - y` (gradient) and `1` (hessian).
    SquaredLoss,
    /// Quantile Loss for regression.
    /// Minimizes `quantile * |y - yhat|` when `y >= yhat`, and `(1 - quantile) * |y - yhat|` when `y < yhat`.
    /// Useful for predicting specific percentiles (e.g., median with quantile=0.5).
    QuantileLoss {
        /// Target quantile (e.g., 0.5 for median).
        quantile: Option<f64>,
    },
    /// Huber Loss for robust regression.
    /// A combination of Squared Loss (near 0) and Absolute Loss (far from 0).
    /// Effectively ignores outliers beyond `delta`.
    HuberLoss {
        /// The threshold where the loss function transitions from quadratic to linear.
        delta: Option<f64>,
    },
    /// Adaptive Huber Loss for robust regression.
    /// Automatically adjusts `delta` based on the quantile of the absolute error distribution.
    AdaptiveHuberLoss {
        /// Quantile used to determine the adaptive delta (e.g., 0.95).
        quantile: Option<f64>,
    },
    /// ListNet Loss for Learning-to-Rank.
    /// Optimizes for list-wise ranking metrics.
    /// Requires `group` parameter in `fit` to define query groups.
    ListNetLoss,
    PoissonLoss,
    GammaLoss,
    MapeLoss,
    FairLoss {
        c: Option<f64>,
    },
    TweedieLoss {
        p: Option<f64>,
    },
    SquaredLogLoss,
    /// CrossEntropyLoss for continuous targets in [0, 1]. Identical computation to LogLoss.
    CrossEntropyLoss,
    /// CrossEntropyLambdaLoss for an alternative formulation of cross entropy.
    CrossEntropyLambdaLoss,
    /// Absolute Loss for L1 regression.
    AbsoluteLoss,
    /// Hinge Loss for binary classification where targets are 0 or 1.
    HingeLoss,
    /// Custom user-defined objective.
    #[serde(with = "objective_custom_serde")]
    Custom(Arc<dyn ObjectiveFunction>),
}

mod objective_custom_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(_: &Arc<dyn ObjectiveFunction>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str("Custom")
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Arc<dyn ObjectiveFunction>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let _: String = Deserialize::deserialize(d)?;
        Ok(Arc::new(SquaredLoss::default()))
    }
}

impl Objective {
    pub fn new_custom<T>(objective: T) -> Self
    where
        T: ObjectiveFunction + 'static,
    {
        Objective::Custom(Arc::new(objective))
    }

    /// Per-sample loss for a single observation (no heap allocation).
    /// Used in the inner tree-growth loop to avoid Vec allocations.
    ///
    /// # Note on Implementation
    /// `loss_single` is implemented here rather than in the `ObjectiveFunction` trait to avoid
    /// the overhead of dynamic dispatch and, more importantly, FFI boundary crossing for
    /// `Custom` (Python) objectives.
    ///
    /// For native Rust objectives (`LogLoss`, `SquaredLoss`, etc.), we perform static dispatch
    /// to their inherent `loss_single` methods, ensuring maximum performance in hot loops.
    ///
    /// For `Custom` objectives, this method panics because they should always rely on `loss` (batch)
    /// evaluation, as indicated by `requires_batch_evaluation()` returning `true`.
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        match self {
            Objective::LogLoss => LogLoss::default().loss_single(y, yhat, sample_weight),
            Objective::BrierLoss => BrierLoss::default().loss_single(y, yhat, sample_weight),
            Objective::SquaredLoss => SquaredLoss::default().loss_single(y, yhat, sample_weight),
            Objective::QuantileLoss { quantile } => {
                QuantileLoss { quantile: *quantile }.loss_single(y, yhat, sample_weight)
            }
            Objective::HuberLoss { delta } => HuberLoss { delta: *delta }.loss_single(y, yhat, sample_weight),
            Objective::AdaptiveHuberLoss { quantile } => {
                AdaptiveHuberLoss { quantile: *quantile }.loss_single(y, yhat, sample_weight)
            }
            Objective::ListNetLoss => ListNetLoss::default().loss_single(y, yhat, sample_weight),
            Objective::PoissonLoss => PoissonLoss::default().loss_single(y, yhat, sample_weight),
            Objective::GammaLoss => GammaLoss::default().loss_single(y, yhat, sample_weight),
            Objective::MapeLoss => MapeLoss::default().loss_single(y, yhat, sample_weight),
            Objective::FairLoss { c } => FairLoss { c: *c }.loss_single(y, yhat, sample_weight),
            Objective::TweedieLoss { p } => TweedieLoss { p: *p }.loss_single(y, yhat, sample_weight),
            Objective::SquaredLogLoss => SquaredLogLoss::default().loss_single(y, yhat, sample_weight),
            Objective::CrossEntropyLoss => CrossEntropyLoss::default().loss_single(y, yhat, sample_weight),
            Objective::CrossEntropyLambdaLoss => CrossEntropyLambdaLoss::default().loss_single(y, yhat, sample_weight),
            Objective::AbsoluteLoss => AbsoluteLoss::default().loss_single(y, yhat, sample_weight),
            Objective::HingeLoss => HingeLoss::default().loss_single(y, yhat, sample_weight),
            Objective::Custom(_) => {
                panic!("loss_single should not be called for Custom objectives. Use batch loss instead.")
            }
        }
    }
}

/// Dispatch a method call through the `Objective` enum to the concrete loss.
macro_rules! dispatch {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        match $self {
            Objective::LogLoss => LogLoss::default().$method($($arg),*),
            Objective::BrierLoss => BrierLoss::default().$method($($arg),*),
            Objective::SquaredLoss => SquaredLoss::default().$method($($arg),*),
            Objective::QuantileLoss { quantile } => {
                QuantileLoss { quantile: *quantile }.$method($($arg),*)
            }
            Objective::HuberLoss { delta } => {
                HuberLoss { delta: *delta }.$method($($arg),*)
            }
            Objective::AdaptiveHuberLoss { quantile } => {
                AdaptiveHuberLoss { quantile: *quantile }.$method($($arg),*)
            }
            Objective::ListNetLoss => ListNetLoss::default().$method($($arg),*),
            Objective::PoissonLoss => PoissonLoss::default().$method($($arg),*),
            Objective::GammaLoss => GammaLoss::default().$method($($arg),*),
            Objective::MapeLoss => MapeLoss::default().$method($($arg),*),
            Objective::FairLoss { c } => FairLoss { c: *c }.$method($($arg),*),
            Objective::TweedieLoss { p } => TweedieLoss { p: *p }.$method($($arg),*),
            Objective::SquaredLogLoss => SquaredLogLoss::default().$method($($arg),*),
            Objective::CrossEntropyLoss => CrossEntropyLoss::default().$method($($arg),*),
            Objective::CrossEntropyLambdaLoss => CrossEntropyLambdaLoss::default().$method($($arg),*),
            Objective::AbsoluteLoss => AbsoluteLoss::default().$method($($arg),*),
            Objective::HingeLoss => HingeLoss::default().$method($($arg),*),
            Objective::Custom(arc) => arc.$method($($arg),*),
        }
    };
}

impl ObjectiveFunction for Objective {
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> Vec<f32> {
        dispatch!(self, loss(y, yhat, sample_weight, group))
    }

    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        dispatch!(self, gradient(y, yhat, sample_weight, group))
    }

    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> f64 {
        dispatch!(self, initial_value(y, sample_weight, group))
    }

    fn default_metric(&self) -> Metric {
        dispatch!(self, default_metric())
    }

    fn gradient_and_loss(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>, Vec<f32>) {
        dispatch!(self, gradient_and_loss(y, yhat, sample_weight, group))
    }

    fn gradient_and_loss_into(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        grad: &mut [f32],
        hess: &mut Option<Vec<f32>>,
        loss: &mut [f32],
    ) {
        dispatch!(
            self,
            gradient_and_loss_into(y, yhat, sample_weight, group, grad, hess, loss)
        )
    }

    fn requires_batch_evaluation(&self) -> bool {
        dispatch!(self, requires_batch_evaluation())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objective::Objective;

    // Common data used across tests
    static Y: &[f64] = &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    static YHAT1: &[f64] = &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    static YHAT2: &[f64] = &[0.0, 0.0, -1.0, 1.0, 0.0, 1.0];

    // new helper function for the tests
    fn sum_loss(obj: &Objective, yhat: &[f64]) -> f32 {
        obj.loss(Y, yhat, None, None).iter().copied().sum()
    }

    fn sum_grad(obj: &Objective, yhat: &[f64]) -> f32 {
        let (g, _) = obj.gradient(Y, yhat, None, None);
        g.iter().copied().sum()
    }

    // actual tests
    #[test]
    fn test_logloss_loss() {
        let objective_function = Objective::LogLoss;
        assert!(sum_loss(&objective_function, YHAT1) < sum_loss(&objective_function, YHAT2));
    }

    #[test]
    fn test_logloss_grad() {
        let objective_function = Objective::LogLoss;
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    #[test]
    fn test_logloss_init() {
        let objective_function = Objective::LogLoss;
        assert_eq!(objective_function.initial_value(Y, None, None), 0.0);

        let all_ones = vec![1.0; 6];
        assert_eq!(Objective::LogLoss.initial_value(&all_ones, None, None), f64::INFINITY);

        let all_zeros = vec![0.0; 6];
        assert_eq!(
            Objective::LogLoss.initial_value(&all_zeros, None, None),
            f64::NEG_INFINITY
        );

        let mixed = &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let expected = f64::ln(2.0 / 4.0);
        assert_eq!(Objective::LogLoss.initial_value(mixed, None, None), expected);
    }

    #[test]
    fn test_mse_init() {
        let objective_function = Objective::SquaredLoss;
        assert_eq!(objective_function.initial_value(Y, None, None), 0.5);

        let all_ones = vec![1.0; 6];
        assert_eq!(Objective::SquaredLoss.initial_value(&all_ones, None, None), 1.0);

        let all_minus = vec![-1.0; 6];
        assert_eq!(Objective::SquaredLoss.initial_value(&all_minus, None, None), -1.0);

        let mixed = &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        assert_eq!(Objective::SquaredLoss.initial_value(mixed, None, None), 0.0);
    }

    #[test]
    fn test_quantile_init() {
        let weights = &[0.0, 0.5, 1.0, 0.3, 0.5];
        let y_vals = &[1.0, 2.0, 9.0, 3.2, 4.0];

        let objective_function_low = Objective::QuantileLoss { quantile: Some(0.1) };
        assert_eq!(objective_function_low.initial_value(y_vals, Some(weights), None), 2.0);

        let objective_function_high = Objective::QuantileLoss { quantile: Some(0.9) };
        assert_eq!(objective_function_high.initial_value(y_vals, Some(weights), None), 9.0);
    }

    #[test]
    fn test_adaptive_huberloss_loss_and_grad() {
        let objective_function = Objective::AdaptiveHuberLoss { quantile: Some(0.5) };
        assert!(sum_loss(&objective_function, YHAT1) > sum_loss(&objective_function, YHAT2));
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    #[test]
    fn test_huberloss_loss_and_grad() {
        let objective_function = Objective::HuberLoss { delta: Some(1.0) };
        assert!(sum_loss(&objective_function, YHAT1) > sum_loss(&objective_function, YHAT2));
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    static Y_RANK: &[f64] = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    static YHAT1_RANK: &[f64] = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    static YHAT2_RANK: &[f64] = &[3.0, 2.0, 1.0, 3.0, 2.0, 1.0];
    static YHAT3_RANK: &[f64] = &[4.0, 5.0, 6.0, 4.0, 5.0, 6.0]; // NOTE: should be the
    // same as YHAT1_RANK
    static GROUP: &[u64] = &[3, 3];

    fn sum_loss_rank(obj: &Objective, yhat: &[f64]) -> f32 {
        obj.loss(Y_RANK, yhat, None, Some(GROUP)).iter().copied().sum()
    }

    fn sum_grad_rank(obj: &Objective, yhat: &[f64]) -> f32 {
        let (g, _) = obj.gradient(Y_RANK, yhat, None, Some(GROUP));
        g.iter().map(|x| x.abs()).sum()
    }

    #[test]
    fn test_listnet_loss_and_grad() {
        let objective_function = Objective::ListNetLoss;
        let good_loss_sum = sum_loss_rank(&objective_function, YHAT1_RANK);
        let bad_loss_sum = sum_loss_rank(&objective_function, YHAT2_RANK);
        let also_good_loss_sum = sum_loss_rank(&objective_function, YHAT3_RANK);

        let good_grad_sum = sum_grad_rank(&objective_function, YHAT1_RANK);
        let bad_grad_sum = sum_grad_rank(&objective_function, YHAT2_RANK);
        let also_good_grad_sum = sum_grad_rank(&objective_function, YHAT3_RANK);

        assert!(good_loss_sum < bad_loss_sum);
        assert!(good_grad_sum < bad_grad_sum);

        assert!(good_loss_sum == also_good_loss_sum);
        assert!(good_grad_sum == also_good_grad_sum);
    }

    #[test]
    fn test_objective_dispatch_gradient_and_loss() {
        let objectives: Vec<Objective> = vec![
            Objective::LogLoss,
            Objective::BrierLoss,
            Objective::SquaredLoss,
            Objective::QuantileLoss { quantile: Some(0.5) },
            Objective::HuberLoss { delta: Some(1.0) },
            Objective::AdaptiveHuberLoss { quantile: Some(0.5) },
            Objective::PoissonLoss,
            Objective::GammaLoss,
            Objective::MapeLoss,
            Objective::FairLoss { c: Some(1.0) },
            Objective::TweedieLoss { p: Some(1.5) },
            Objective::SquaredLogLoss,
            Objective::CrossEntropyLoss,
            Objective::CrossEntropyLambdaLoss,
            Objective::AbsoluteLoss,
            Objective::HingeLoss,
        ];
        for obj in &objectives {
            let (g, _h, l) = obj.gradient_and_loss(Y, YHAT1, None::<&[f64]>, None::<&[u64]>);
            assert_eq!(g.len(), Y.len());
            assert_eq!(l.len(), Y.len());
        }
        // ListNetLoss needs group
        let (g, _h, l) = Objective::ListNetLoss.gradient_and_loss(Y_RANK, YHAT1_RANK, None::<&[f64]>, Some(GROUP));
        assert_eq!(g.len(), Y_RANK.len());
        assert_eq!(l.len(), Y_RANK.len());
    }

    #[test]
    fn test_objective_dispatch_gradient_and_loss_into() {
        let objectives: Vec<Objective> = vec![
            Objective::LogLoss,
            Objective::BrierLoss,
            Objective::SquaredLoss,
            Objective::QuantileLoss { quantile: Some(0.5) },
            Objective::HuberLoss { delta: Some(1.0) },
            Objective::AdaptiveHuberLoss { quantile: Some(0.5) },
            Objective::PoissonLoss,
            Objective::GammaLoss,
            Objective::MapeLoss,
            Objective::FairLoss { c: Some(1.0) },
            Objective::TweedieLoss { p: Some(1.5) },
            Objective::SquaredLogLoss,
            Objective::CrossEntropyLoss,
            Objective::CrossEntropyLambdaLoss,
            Objective::AbsoluteLoss,
            Objective::HingeLoss,
        ];
        for obj in &objectives {
            let mut grad = vec![0.0_f32; Y.len()];
            let mut hess: Option<Vec<f32>> = None;
            let mut loss = vec![0.0_f32; Y.len()];
            obj.gradient_and_loss_into(
                Y,
                YHAT1,
                None::<&[f64]>,
                None::<&[u64]>,
                &mut grad,
                &mut hess,
                &mut loss,
            );
        }
    }

    #[test]
    fn test_objective_loss_single_all() {
        let objectives: Vec<Objective> = vec![
            Objective::LogLoss,
            Objective::BrierLoss,
            Objective::SquaredLoss,
            Objective::QuantileLoss { quantile: Some(0.5) },
            Objective::HuberLoss { delta: Some(1.0) },
            Objective::AdaptiveHuberLoss { quantile: Some(0.5) },
            Objective::PoissonLoss,
            Objective::GammaLoss,
            Objective::MapeLoss,
            Objective::FairLoss { c: Some(1.0) },
            Objective::TweedieLoss { p: Some(1.5) },
            Objective::SquaredLogLoss,
            Objective::CrossEntropyLoss,
            Objective::CrossEntropyLambdaLoss,
            Objective::AbsoluteLoss,
            Objective::HingeLoss,
        ];
        for obj in &objectives {
            let _ = obj.loss_single(1.0, 0.5, None);
        }
        let l6 = Objective::ListNetLoss.loss_single(1.0, 0.5, None);
        assert_eq!(l6, f32::INFINITY);
    }

    #[test]
    fn test_objective_requires_batch() {
        assert!(!Objective::LogLoss.requires_batch_evaluation());
        assert!(!Objective::SquaredLoss.requires_batch_evaluation());
        assert!(Objective::ListNetLoss.requires_batch_evaluation());
    }

    #[test]
    fn test_objective_dispatch_default_metric() {
        let objectives: Vec<Objective> = vec![
            Objective::LogLoss,
            Objective::BrierLoss,
            Objective::SquaredLoss,
            Objective::QuantileLoss { quantile: Some(0.5) },
            Objective::HuberLoss { delta: Some(1.0) },
            Objective::AdaptiveHuberLoss { quantile: Some(0.5) },
            Objective::PoissonLoss,
            Objective::GammaLoss,
            Objective::MapeLoss,
            Objective::FairLoss { c: Some(1.0) },
            Objective::TweedieLoss { p: Some(1.5) },
            Objective::SquaredLogLoss,
            Objective::CrossEntropyLoss,
            Objective::CrossEntropyLambdaLoss,
            Objective::AbsoluteLoss,
            Objective::HingeLoss,
            Objective::ListNetLoss,
        ];
        for obj in objectives {
            let _ = obj.default_metric();
        }
    }

    #[test]
    fn test_custom_objective() {
        struct MyObj;
        impl ObjectiveFunction for MyObj {
            fn loss(&self, y: &[f64], yhat: &[f64], _sw: Option<&[f64]>, _g: Option<&[u64]>) -> Vec<f32> {
                y.iter().zip(yhat).map(|(y, yh)| (y - yh).powi(2) as f32).collect()
            }
            fn gradient(
                &self,
                y: &[f64],
                yhat: &[f64],
                _sw: Option<&[f64]>,
                _g: Option<&[u64]>,
            ) -> (Vec<f32>, Option<Vec<f32>>) {
                let g = y.iter().zip(yhat).map(|(y, yh)| (yh - y) as f32).collect();
                (g, None)
            }
        }
        let obj = Objective::new_custom(MyObj);
        assert!(obj.requires_batch_evaluation());
        let l = obj.loss(&[1.0], &[0.0], None, None);
        assert_eq!(l[0], 1.0);
    }

    #[test]
    #[should_panic]
    fn test_custom_objective_panic() {
        struct MyObj;
        impl ObjectiveFunction for MyObj {
            fn loss(&self, _: &[f64], _: &[f64], _: Option<&[f64]>, _: Option<&[u64]>) -> Vec<f32> {
                vec![]
            }
            fn gradient(
                &self,
                _: &[f64],
                _: &[f64],
                _: Option<&[f64]>,
                _: Option<&[u64]>,
            ) -> (Vec<f32>, Option<Vec<f32>>) {
                (vec![], None)
            }
        }
        let obj = Objective::new_custom(MyObj);
        obj.loss_single(1.0, 0.0, None);
    }

    #[test]
    fn test_objective_serde() {
        let obj = Objective::LogLoss;
        let s = serde_json::to_string(&obj).unwrap();
        let _: Objective = serde_json::from_str(&s).unwrap();

        // Custom serde
        let custom = Objective::new_custom(SquaredLoss::default());
        let s2 = serde_json::to_string(&custom).unwrap();
        assert!(s2.contains("\"Custom\""));
        let d2: Objective = serde_json::from_str(&s2).unwrap();
        match d2 {
            Objective::Custom(_) => (),
            _ => panic!("Expected Custom objective"),
        }
    }
}
