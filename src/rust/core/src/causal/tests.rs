#[cfg(test)]
mod causal_tests {
    use crate::causal::dml::DMLObjective;
    use crate::causal::fairness::{FairnessObjective, FairnessType};
    use crate::causal::iv::IVBooster;
    use crate::causal::policy::PolicyObjective;
    use crate::causal::uplift::UpliftBooster;
    use crate::data::Matrix;
    use crate::objective_functions::objective::{Objective, ObjectiveFunction};

    // -----------------------------------------------------------------------
    // UpliftBooster
    // -----------------------------------------------------------------------

    #[test]
    fn test_uplift_booster_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = Matrix::new(&data, 4, 1);
        let w = vec![0.0, 1.0, 0.0, 1.0];
        let y = vec![1.0, 3.0, 2.0, 4.0];

        let mut booster = UpliftBooster::new(
            0.5,
            0.5,
            0.5,
            255,
            None,
            None,
            None,
            false,
            f64::NAN,
            true,
            true,
            Default::default(),
            crate::booster::config::MissingNodeTreatment::AssignToParent,
            0,
            42,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        booster.fit(&matrix, &w, &y).expect("Fit failed");

        let preds = booster.predict(&matrix);
        assert_eq!(preds.len(), 4);
        assert!(preds.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_uplift_booster_larger_data() {
        // Synthetic data: outcome = 0.5 * x + treatment_effect * w + noise
        let n = 50;
        let mut x_data = Vec::with_capacity(n);
        let mut w = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let xi = (i as f64) / (n as f64);
            x_data.push(xi);
            let wi = if i % 2 == 0 { 1.0 } else { 0.0 };
            w.push(wi);
            let effect = 0.3; // constant treatment effect
            y.push(0.5 * xi + effect * wi + 0.01 * (i as f64 % 7.0));
        }
        let matrix = Matrix::new(&x_data, n, 1);

        let mut booster = UpliftBooster::new(
            0.3,
            0.3,
            0.3,
            255,
            None,
            None,
            None,
            false,
            f64::NAN,
            true,
            true,
            Default::default(),
            crate::booster::config::MissingNodeTreatment::AssignToParent,
            0,
            42,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        booster.fit(&matrix, &w, &y).expect("Fit failed");
        let preds = booster.predict(&matrix);
        assert_eq!(preds.len(), n);

        // Average predicted effect should be roughly positive
        let avg: f64 = preds.iter().sum::<f64>() / n as f64;
        assert!(avg > -0.5, "Average uplift should be near-positive, got {avg}");
    }

    // -----------------------------------------------------------------------
    // IVBooster
    // -----------------------------------------------------------------------

    #[test]
    fn test_iv_booster_basic() {
        let x_data = vec![1.0, 1.0, 1.0, 1.0];
        let x_matrix = Matrix::new(&x_data, 4, 1);

        let z_data = vec![0.0, 1.0, 0.0, 1.0];
        let z_matrix = Matrix::new(&z_data, 4, 1);

        let w = vec![0.0, 1.0, 0.0, 1.0];
        let y = vec![0.0, 2.0, 0.0, 2.0];

        let mut booster = IVBooster::new(
            Objective::SquaredLoss,
            Objective::SquaredLoss,
            0.5,
            0.5,
            255,
            None,
            None,
            None,
            false,
            f64::NAN,
            true,
            true,
            Default::default(),
            crate::booster::config::MissingNodeTreatment::AssignToParent,
            0,
            42,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        booster.fit(&x_matrix, &z_matrix, &y, &w).expect("Fit failed");

        let preds = booster.predict(&x_matrix, &[1.0; 4]);
        assert_eq!(preds.len(), 4);
        assert!(preds.iter().all(|x| x.is_finite()));
    }

    // -----------------------------------------------------------------------
    // PolicyObjective — IPW
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_objective_ipw() {
        let treatment = vec![1, 1];
        let propensity = vec![0.5, 0.5];
        let obj = PolicyObjective::new(treatment, propensity);

        let y = vec![10.0, -10.0];
        let yhat = vec![0.0, 0.0]; // sigmoid(0) = 0.5

        let (grad, hess) = obj.gradient(&y, &yhat, None, None);
        assert_eq!(grad.len(), 2);
        assert!(hess.is_some());

        // Sample 1: Y=10, W=1, p=0.5 → pseudo = 10/0.5 = 20 → target=1
        // grad = 20 * (0.5 - 1) = -10 → negative (push score up)
        assert!(grad[0] < 0.0, "Grad should be negative for positive reward");

        // Sample 2: Y=-10, W=1, p=0.5 → pseudo = -10/0.5 - 0 = -20 → target=0
        // grad = 20 * (0.5 - 0) = 10 → positive (push score down)
        assert!(grad[1] > 0.0, "Grad should be positive for negative reward");
    }

    // -----------------------------------------------------------------------
    // PolicyObjective — AIPW (Doubly Robust)
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_objective_aipw() {
        let treatment = vec![1, 0, 1, 0];
        let propensity = vec![0.5, 0.5, 0.5, 0.5];
        let mu_hat = vec![5.0, 5.0, 5.0, 5.0]; // baseline prediction

        let obj = PolicyObjective::new_aipw(treatment, propensity, mu_hat);

        let y = vec![10.0, 2.0, 8.0, 3.0];
        let yhat = vec![0.0; 4];

        let (grad, hess) = obj.gradient(&y, &yhat, None, None);
        assert_eq!(grad.len(), 4);
        assert!(hess.is_some());

        let loss = obj.loss(&y, &yhat, None, None);
        assert_eq!(loss.len(), 4);
    }

    // -----------------------------------------------------------------------
    // FairnessObjective — Demographic Parity
    // -----------------------------------------------------------------------

    #[test]
    fn test_fairness_objective_demographic_parity() {
        let sensitive = vec![1, 1, 0, 0];
        let obj = FairnessObjective::new(sensitive, 1.0);

        let y = vec![1.0, 0.0, 1.0, 0.0];
        let yhat = vec![1.38, -1.38, 1.38, -1.38]; // ~0.8, ~0.2

        let (grad, hess) = obj.gradient(&y, &yhat, None, None);

        // When groups have equal mean predictions, fairness gradient is ~0.
        // Standard LogLoss grad for p≈0.8, y=1: 0.8-1 = -0.2
        assert!((grad[0] + 0.2).abs() < 0.02);
        assert!(hess.is_some());
    }

    // -----------------------------------------------------------------------
    // FairnessObjective — Equalized Odds
    // -----------------------------------------------------------------------

    #[test]
    fn test_fairness_objective_equalized_odds() {
        let sensitive = vec![1, 1, 0, 0, 1, 0];
        let obj = FairnessObjective::with_type(sensitive, 1.0, FairnessType::EqualizedOdds);

        let y = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let yhat = vec![1.0, -1.0, 1.0, -1.0, 2.0, -2.0];

        let (grad, hess) = obj.gradient(&y, &yhat, None, None);
        assert_eq!(grad.len(), 6);
        assert!(hess.is_some());

        let loss = obj.loss(&y, &yhat, None, None);
        assert_eq!(loss.len(), 6);
        // All losses should be non-negative
        assert!(loss.iter().all(|&l| l >= 0.0));
    }

    // -----------------------------------------------------------------------
    // DMLObjective
    // -----------------------------------------------------------------------

    #[test]
    fn test_dml_objective_gradient() {
        // y_res = [2, 4, 6], w_res = [1, 2, 3]
        // True theta ~ y_res / w_res = 2 for all.
        let y_res = vec![2.0, 4.0, 6.0];
        let w_res = vec![1.0, 2.0, 3.0];
        let obj = DMLObjective::new(y_res, w_res);

        // If theta = 2.0 (correct), gradient should be ~0.
        let y_dummy = vec![0.0; 3]; // unused by DML
        let yhat = vec![2.0, 2.0, 2.0];

        let (grad, hess) = obj.gradient(&y_dummy, &yhat, None, None);
        for g in &grad {
            assert!(g.abs() < 1e-5, "Gradient at true theta should be near zero, got {g}");
        }
        // Hessian = w_res^2
        let hess = hess.unwrap();
        assert!((hess[0] - 1.0).abs() < 1e-5);
        assert!((hess[1] - 4.0).abs() < 1e-5);
        assert!((hess[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_dml_objective_initial_value() {
        // OLS estimate: sum(y_res * w_res) / sum(w_res^2)
        // = (2*1 + 4*2 + 6*3) / (1 + 4 + 9) = (2+8+18)/14 = 28/14 = 2.0
        let y_res = vec![2.0, 4.0, 6.0];
        let w_res = vec![1.0, 2.0, 3.0];
        let obj = DMLObjective::new(y_res, w_res);

        let init = obj.initial_value(&[0.0; 3], None, None);
        assert!((init - 2.0).abs() < 1e-10, "Initial value should be 2.0, got {init}");
    }

    #[test]
    fn test_dml_objective_loss() {
        let y_res = vec![2.0, 4.0];
        let w_res = vec![1.0, 2.0];
        let obj = DMLObjective::new(y_res, w_res);

        // theta = 0 → loss = (2-0)^2 + (4-0)^2 = [4, 16]
        let loss = obj.loss(&[0.0; 2], &[0.0, 0.0], None, None);
        assert!((loss[0] - 4.0).abs() < 1e-5);
        assert!((loss[1] - 16.0).abs() < 1e-5);

        // theta = 2 → loss = (2-2*1)^2 + (4-2*2)^2 = [0, 0]
        let loss = obj.loss(&[0.0; 2], &[2.0, 2.0], None, None);
        assert!(loss[0].abs() < 1e-5);
        assert!(loss[1].abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_dml_objective_near_zero_weight() {
        // When w_res ≈ 0, hessian is floored to prevent blow-up.
        let y_res = vec![1.0];
        let w_res = vec![1e-10];
        let obj = DMLObjective::new(y_res, w_res);

        let (grad, hess) = obj.gradient(&[0.0], &[0.0], None, None);
        let hess = hess.unwrap();
        assert!(hess[0] >= 1e-6, "Hessian should be floored");
        assert!(grad[0].is_finite());
    }

    #[test]
    fn test_policy_extreme_propensity() {
        // Propensity near 0 or 1 should be clipped.
        let treatment = vec![1, 0];
        let propensity = vec![0.0001, 0.9999];
        let obj = PolicyObjective::new(treatment, propensity);

        let y = vec![1.0, 1.0];
        let yhat = vec![0.0, 0.0];

        let (grad, _) = obj.gradient(&y, &yhat, None, None);
        assert!(grad.iter().all(|g| g.is_finite()), "Gradients must be finite");
    }
}
