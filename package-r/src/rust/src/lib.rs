use extendr_api::prelude::*;
use perpetual_rs::booster::config::{BoosterConfig, BoosterIO, ContributionsMethod, MissingNodeTreatment};
use perpetual_rs::objective_functions::Objective;
use perpetual_rs::Matrix;
use perpetual_rs::MultiOutputBooster;
use perpetual_rs::PerpetualBooster as CratePerpetualBooster;

enum InternalBooster {
    Single(CratePerpetualBooster),
    Multi(MultiOutputBooster),
}

#[extendr]
pub struct PerpetualBooster {
    internal: Option<InternalBooster>,
    config: BoosterConfig,
    classes: Vec<f64>,
}

#[extendr]
impl PerpetualBooster {
    pub fn new(
        objective: Option<&str>,
        budget: Option<f64>,
        max_bin: Option<i32>,
        num_threads: Option<i32>,
        missing: Option<f64>,
        allow_missing_splits: Option<bool>,
        create_missing_branch: Option<bool>,
        missing_node_treatment: Option<&str>,
        log_iterations: Option<i32>,
        quantile: Option<f64>,
        reset: Option<bool>,
        timeout: Option<f64>,
        iteration_limit: Option<i32>,
        memory_limit: Option<f64>,
        stopping_rounds: Option<i32>,
        seed: Option<i32>,
    ) -> Self {
        let mut config = BoosterConfig::default();

        if let Some(obj) = objective {
            if let Ok(obj_enum) = serde_plain::from_str::<Objective>(obj) {
                config.objective = obj_enum;
            } else if !obj.is_empty() {
                rprintln!("Warning: Unknown objective '{}'. Using default LogLoss.", obj);
            }
        }

        if let Some(b) = budget {
            config.budget = b as f32;
        }
        if let Some(mb) = max_bin {
            config.max_bin = mb as u16;
        }
        if let Some(nt) = num_threads {
            if nt > 0 {
                config.num_threads = Some(nt as usize);
            }
        }

        if let Some(val) = missing {
            config.missing = val;
        }
        if let Some(val) = allow_missing_splits {
            config.allow_missing_splits = val;
        }
        if let Some(val) = create_missing_branch {
            config.create_missing_branch = val;
        }

        if let Some(treatment) = missing_node_treatment {
            if let Ok(t) = serde_plain::from_str::<MissingNodeTreatment>(treatment) {
                config.missing_node_treatment = t;
            }
        }

        if let Some(val) = log_iterations {
            config.log_iterations = val as usize;
        }
        if let Some(val) = quantile {
            config.quantile = Some(val);
        }
        if let Some(val) = reset {
            config.reset = Some(val);
        }
        if let Some(val) = timeout {
            config.timeout = Some(val as f32);
        }
        if let Some(val) = iteration_limit {
            config.iteration_limit = Some(val as usize);
        }
        if let Some(val) = memory_limit {
            config.memory_limit = Some(val as f32);
        }
        if let Some(val) = stopping_rounds {
            config.stopping_rounds = Some(val as usize);
        }
        if let Some(val) = seed {
            config.seed = val as u64;
        }

        Self {
            internal: None,
            config,
            classes: Vec::new(),
        }
    }

    pub fn fit(&mut self, flat_data: Vec<f64>, rows: i32, cols: i32, y: Vec<f64>) {
        let rows = rows as usize;
        let cols = cols as usize;
        let matrix = Matrix::new(&flat_data, rows, cols);

        // Detect unique classes
        let mut unique_classes: Vec<f64> = y.iter().filter(|v| !v.is_nan()).cloned().collect();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_classes.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
        self.classes = unique_classes;

        if self.classes.len() > 2 && matches!(self.config.objective, Objective::LogLoss) {
            // Multiclass classification (One-vs-Rest)
            let n_classes = self.classes.len();
            let mut y_matrix_data = vec![0.0; rows * n_classes];
            for (i, &val) in y.iter().enumerate() {
                if let Some(pos) = self.classes.iter().position(|&c| (c - val).abs() < f64::EPSILON) {
                    y_matrix_data[pos * rows + i] = 1.0;
                }
            }
            let y_matrix = Matrix::new(&y_matrix_data, rows, n_classes);

            let mut multi_booster = MultiOutputBooster::new(
                n_classes,
                self.config.objective.clone(),
                self.config.budget,
                self.config.max_bin,
                self.config.num_threads,
                self.config.monotone_constraints.clone(),
                self.config.force_children_to_bound_parent,
                self.config.missing,
                self.config.allow_missing_splits,
                self.config.create_missing_branch,
                self.config.terminate_missing_features.clone(),
                self.config.missing_node_treatment,
                self.config.log_iterations,
                self.config.seed,
                self.config.quantile,
                self.config.reset,
                self.config.categorical_features.clone(),
                self.config.timeout,
                self.config.iteration_limit,
                self.config.memory_limit,
                self.config.stopping_rounds,
            )
            .unwrap();

            multi_booster.fit(&matrix, &y_matrix, None, None).unwrap();

            // Store classes in metadata for each booster
            let classes_str = self.classes.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
            for b_ in &mut multi_booster.boosters {
                b_.metadata.insert("classes".to_string(), classes_str.clone());
            }

            self.internal = Some(InternalBooster::Multi(multi_booster));
        } else {
            // Single output (Regression or Binary Classification)
            let mut booster = CratePerpetualBooster::new(
                self.config.objective.clone(),
                self.config.budget,
                f64::NAN, // base_score will be calculated
                self.config.max_bin,
                self.config.num_threads,
                self.config.monotone_constraints.clone(),
                self.config.force_children_to_bound_parent,
                self.config.missing,
                self.config.allow_missing_splits,
                self.config.create_missing_branch,
                self.config.terminate_missing_features.clone(),
                self.config.missing_node_treatment,
                self.config.log_iterations,
                self.config.seed,
                self.config.quantile,
                self.config.reset,
                self.config.categorical_features.clone(),
                self.config.timeout,
                self.config.iteration_limit,
                self.config.memory_limit,
                self.config.stopping_rounds,
            )
            .unwrap();

            booster.fit(&matrix, &y, None, None::<&[u64]>).unwrap();

            // Store classes in metadata
            if !self.classes.is_empty() {
                let classes_str = self.classes.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
                booster.metadata.insert("classes".to_string(), classes_str);
            }

            self.internal = Some(InternalBooster::Single(booster));
        }
    }

    pub fn predict(&self, flat_data: Vec<f64>, rows: i32, cols: i32) -> Vec<f64> {
        let rows = rows as usize;
        let cols = cols as usize;
        let matrix = Matrix::new(&flat_data, rows, cols);

        match &self.internal {
            Some(InternalBooster::Single(b)) => b.predict(&matrix, true),
            Some(InternalBooster::Multi(b)) => b.predict(&matrix, true),
            None => panic!("Booster not fitted"),
        }
    }

    pub fn predict_proba(&self, flat_data: Vec<f64>, rows: i32, cols: i32) -> Vec<f64> {
        let rows = rows as usize;
        let cols = cols as usize;
        let matrix = Matrix::new(&flat_data, rows, cols);

        match &self.internal {
            Some(InternalBooster::Single(b)) => b.predict_proba(&matrix, true),
            Some(InternalBooster::Multi(b)) => b.predict_proba(&matrix, true),
            None => panic!("Booster not fitted"),
        }
    }

    pub fn save_booster(&self, path: &str) {
        match &self.internal {
            Some(InternalBooster::Single(b)) => b.save_booster(path).unwrap(),
            Some(InternalBooster::Multi(b)) => b.save_booster(path).unwrap(),
            None => panic!("Booster not fitted"),
        }
    }

    pub fn load_booster(path: &str) -> Self {
        let json_str = std::fs::read_to_string(path).unwrap();
        let mut classes = Vec::new();

        let internal = if json_str.contains("\"boosters\":") {
            let b = MultiOutputBooster::from_json(&json_str).unwrap();
            // Try to extract classes from the first booster's metadata
            if let Some(metadata) = b.boosters.first().map(|b_| &b_.metadata) {
                if let Some(c_str) = metadata.get("classes") {
                    classes = c_str.split(',').filter_map(|s| s.parse::<f64>().ok()).collect();
                }
            }
            Some(InternalBooster::Multi(b))
        } else {
            let b = CratePerpetualBooster::from_json(&json_str).unwrap();
            if let Some(c_str) = b.metadata.get("classes") {
                classes = c_str.split(',').filter_map(|s| s.parse::<f64>().ok()).collect();
            }
            Some(InternalBooster::Single(b))
        };

        let config = match internal.as_ref().unwrap() {
            InternalBooster::Single(b) => b.cfg.clone(),
            InternalBooster::Multi(b) => b.cfg.clone(),
        };

        Self {
            internal,
            config,
            classes,
        }
    }

    pub fn json_dump(&self) -> String {
        match &self.internal {
            Some(InternalBooster::Single(b)) => b.json_dump().unwrap(),
            Some(InternalBooster::Multi(b)) => b.json_dump().unwrap(),
            None => panic!("Booster not fitted"),
        }
    }

    pub fn number_of_trees(&self) -> i32 {
        match &self.internal {
            Some(InternalBooster::Single(b)) => b.get_prediction_trees().len() as i32,
            Some(InternalBooster::Multi(b)) => b
                .boosters
                .iter()
                .map(|b_| b_.get_prediction_trees().len())
                .sum::<usize>() as i32,
            None => 0,
        }
    }

    pub fn base_score(&self) -> f64 {
        match &self.internal {
            Some(InternalBooster::Single(b)) => b.base_score,
            Some(InternalBooster::Multi(b)) => b.boosters.first().map(|b_| b_.base_score).unwrap_or(f64::NAN),
            None => f64::NAN,
        }
    }

    pub fn predict_contributions(&self, flat_data: Vec<f64>, rows: i32, cols: i32, method: &str) -> Vec<f64> {
        let rows = rows as usize;
        let cols = cols as usize;
        let matrix = Matrix::new(&flat_data, rows, cols);

        let method_enum = match method.to_lowercase().as_str() {
            "weight" => ContributionsMethod::Weight,
            "average" => ContributionsMethod::Average,
            "branchdifference" | "branch_difference" => ContributionsMethod::BranchDifference,
            "midpointdifference" | "midpoint_difference" => ContributionsMethod::MidpointDifference,
            "modedifference" | "mode_difference" => ContributionsMethod::ModeDifference,
            "probabilitychange" | "probability_change" => ContributionsMethod::ProbabilityChange,
            "shapley" => ContributionsMethod::Shapley,
            _ => ContributionsMethod::Average,
        };

        match &self.internal {
            Some(InternalBooster::Single(b)) => b.predict_contributions(&matrix, method_enum, true),
            Some(InternalBooster::Multi(b)) => b.predict_contributions(&matrix, method_enum, true),
            None => panic!("Booster not fitted"),
        }
    }

    pub fn calibrate(
        &mut self,
        flat_data: Vec<f64>,
        rows: i32,
        cols: i32,
        y: Vec<f64>,
        flat_data_cal: Vec<f64>,
        rows_cal: i32,
        cols_cal: i32,
        y_cal: Vec<f64>,
        alpha: Vec<f64>,
    ) {
        let rows = rows as usize;
        let cols = cols as usize;
        let matrix = Matrix::new(&flat_data, rows, cols);

        let rows_cal = rows_cal as usize;
        let cols_cal = cols_cal as usize;
        let matrix_cal = Matrix::new(&flat_data_cal, rows_cal, cols_cal);

        let data_cal = (matrix_cal, y_cal.as_slice(), alpha.as_slice());

        match &mut self.internal {
            Some(InternalBooster::Single(b)) => {
                b.calibrate(&matrix, &y, None, None, data_cal).unwrap();
            }
            _ => panic!("Calibration is only supported for single-output boosters currently"),
        }
    }

    pub fn predict_intervals(&self, flat_data: Vec<f64>, rows: i32, cols: i32) -> List {
        let rows = rows as usize;
        let cols = cols as usize;
        let matrix = Matrix::new(&flat_data, rows, cols);

        match &self.internal {
            Some(InternalBooster::Single(b)) => {
                let intervals = b.predict_intervals(&matrix, true);
                let mut r_list = List::new(intervals.len());
                for (i, (_, preds)) in intervals.into_iter().enumerate() {
                    let mut alpha_list = List::new(preds.len());
                    for (j, p) in preds.into_iter().enumerate() {
                        alpha_list.set_elt(j, p.into_robj()).unwrap();
                    }
                    r_list.set_elt(i, alpha_list.into_robj()).unwrap();
                }
                r_list
            }
            _ => panic!("Prediction intervals are not supported for this booster type"),
        }
    }

    pub fn calculate_feature_importance(&self, method: &str, normalize: bool) -> List {
        use perpetual_rs::booster::config::ImportanceMethod;
        let method_enum = match method.to_lowercase().as_str() {
            "weight" => ImportanceMethod::Weight,
            "gain" => ImportanceMethod::Gain,
            "totalgain" | "total_gain" => ImportanceMethod::TotalGain,
            "cover" => ImportanceMethod::Cover,
            "totalcover" | "total_cover" => ImportanceMethod::TotalCover,
            _ => ImportanceMethod::Gain,
        };

        let importance = match &self.internal {
            Some(InternalBooster::Single(b)) => b.calculate_feature_importance(method_enum, normalize),
            Some(InternalBooster::Multi(b)) => b.calculate_feature_importance(method_enum, normalize),
            None => panic!("Booster not fitted"),
        };

        // Convert HashMap<usize, f32> to R List (named numeric vector would be better but list is easier for now)
        let mut names = Vec::new();
        let mut values = Vec::new();
        for (idx, val) in importance {
            names.push(idx.to_string());
            values.push(val as f64);
        }

        // Since we want a named numeric vector in R, we return a list and convert in R or try to return a RealVector
        let mut r_list = List::new(values.len());
        for (i, v) in values.into_iter().enumerate() {
            r_list.set_elt(i, v.into_robj()).unwrap();
        }
        r_list.set_names(names).unwrap();
        r_list
    }

    pub fn get_classes(&self) -> Vec<f64> {
        self.classes.clone()
    }

    pub fn get_objective(&self) -> &'static str {
        match self.config.objective {
            Objective::LogLoss => "LogLoss",
            Objective::SquaredLoss => "SquaredLoss",
            Objective::QuantileLoss { .. } => "QuantileLoss",
            Objective::HuberLoss { .. } => "HuberLoss",
            Objective::AdaptiveHuberLoss { .. } => "AdaptiveHuberLoss",
            Objective::ListNetLoss => "ListNetLoss",
            Objective::Custom(_) => "Custom",
        }
    }
}

#[extendr]
pub fn test_binding() -> &'static str {
    "Hello from New Rust V3!"
}

#[extendr]
pub fn rust_get_classes(booster: &PerpetualBooster) -> Vec<f64> {
    booster.get_classes()
}

#[extendr]
pub fn rust_get_objective(booster: &PerpetualBooster) -> &'static str {
    booster.get_objective()
}

extendr_module! {
    mod perpetual_rust;
    impl PerpetualBooster;
    fn test_binding;
    fn rust_get_classes;
    fn rust_get_objective;
}
