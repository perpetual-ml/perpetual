use extendr_api::prelude::*;
use perpetual_rs::booster::config::BoosterIO;
use perpetual_rs::booster::config::MissingNodeTreatment;
use perpetual_rs::objective_functions::Objective;
use perpetual_rs::Matrix;
use perpetual_rs::PerpetualBooster as CratePerpetualBooster;

struct PerpetualBooster {
    booster: CratePerpetualBooster,
}

#[extendr]
impl PerpetualBooster {
    fn new(
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
    ) -> Self {
        let mut booster = CratePerpetualBooster::default();

        if let Some(obj) = objective {
            if let Ok(obj_enum) = serde_plain::from_str::<Objective>(obj) {
                booster = booster.set_objective(obj_enum);
            } else {
                rprintln!("Warning: Unknown objective '{}'. Using default.", obj);
            }
        }

        if let Some(b) = budget {
            booster = booster.set_budget(b as f32);
        }

        if let Some(mb) = max_bin {
            booster = booster.set_max_bin(mb as u16);
        }

        if let Some(nt) = num_threads {
            if nt > 0 {
                booster = booster.set_num_threads(Some(nt as usize));
            }
        }

        if let Some(val) = missing {
            booster = booster.set_missing(val);
        }
        if let Some(val) = allow_missing_splits {
            booster = booster.set_allow_missing_splits(val);
        }
        if let Some(val) = create_missing_branch {
            booster = booster.set_create_missing_branch(val);
        }

        if let Some(treatment) = missing_node_treatment {
            if let Ok(t) = serde_plain::from_str::<MissingNodeTreatment>(treatment) {
                booster = booster.set_missing_node_treatment(t);
            }
        }

        if let Some(val) = log_iterations {
            booster = booster.set_log_iterations(val as usize);
        }
        if let Some(val) = quantile {
            booster = booster.set_quantile(Some(val));
        }
        if let Some(val) = reset {
            booster = booster.set_reset(Some(val));
        }
        if let Some(val) = timeout {
            booster = booster.set_timeout(Some(val as f32));
        }
        if let Some(val) = iteration_limit {
            booster = booster.set_iteration_limit(Some(val as usize));
        }
        if let Some(val) = memory_limit {
            booster = booster.set_memory_limit(Some(val as f32));
        }
        if let Some(val) = stopping_rounds {
            booster = booster.set_stopping_rounds(Some(val as usize));
        }

        Self { booster }
    }

    fn fit(&mut self, flat_data: Vec<f64>, rows: i32, cols: i32, y: Vec<f64>) {
        let rows = rows as usize;
        let cols = cols as usize;
        let matrix = Matrix::new(&flat_data, rows, cols);

        match self.booster.fit(&matrix, &y, None, None::<&[u64]>) {
            Ok(_) => (),
            Err(e) => panic!("Fit failed: {}", e),
        }
    }

    fn predict(&self, flat_data: Vec<f64>, rows: i32, cols: i32) -> Vec<f64> {
        let rows = rows as usize;
        let cols = cols as usize;
        let matrix = Matrix::new(&flat_data, rows, cols);
        self.booster.predict(&matrix, true)
    }

    fn predict_proba(&self, flat_data: Vec<f64>, rows: i32, cols: i32) -> Vec<f64> {
        let rows = rows as usize;
        let cols = cols as usize;
        let matrix = Matrix::new(&flat_data, rows, cols);
        self.booster.predict_proba(&matrix, true)
    }

    fn save_booster(&self, path: &str) {
        self.booster.save_booster(path).unwrap();
    }

    fn load_booster(path: &str) -> Self {
        let booster = CratePerpetualBooster::load_booster(path).unwrap();
        Self { booster }
    }

    fn json_dump(&self) -> String {
        self.booster.json_dump().unwrap()
    }

    fn number_of_trees(&self) -> i32 {
        self.booster.get_prediction_trees().len() as i32
    }

    fn base_score(&self) -> f64 {
        self.booster.base_score
    }
}

#[extendr]
fn test_binding() -> &'static str {
    "Hello from Rust!"
}

extendr_module! {
    mod perpetual_impl;
    impl PerpetualBooster;
    fn test_binding;
}
