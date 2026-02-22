use perpetual::objective::Objective;
use perpetual::{CalibrationMethod, ColumnarMatrix, PerpetualBooster};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, serde::Deserialize)]
struct HeartDiseaseRow {
    #[serde(rename = "id")]
    id: u32,
    #[serde(rename = "Age")]
    age: f64,
    #[serde(rename = "Sex")]
    sex: String,
    #[serde(rename = "Chest pain type")]
    chest_pain_type: String,
    #[serde(rename = "BP")]
    bp: f64,
    #[serde(rename = "Cholesterol")]
    cholesterol: f64,
    #[serde(rename = "FBS over 120")]
    fbs_over_120: String,
    #[serde(rename = "EKG results")]
    ekg_results: String,
    #[serde(rename = "Max HR")]
    max_hr: f64,
    #[serde(rename = "Exercise angina")]
    exercise_angina: String,
    #[serde(rename = "ST depression")]
    st_depression: f64,
    #[serde(rename = "Slope of ST")]
    slope_of_st: String,
    #[serde(rename = "Number of vessels fluro")]
    vessels: f64,
    #[serde(rename = "Thallium")]
    thallium: String,
    #[serde(rename = "Heart Disease")]
    #[serde(default)]
    heart_disease: Option<String>,
}

fn main() {
    let _ = env_logger::builder().filter_level(log::LevelFilter::Info).try_init();
    println!("Starting Predicting Heart Disease Solution Script (Kaggle S6E2)...");

    let train_path = "examples/kaggle/predicting_heart_disease/train.csv";
    let test_path = "examples/kaggle/predicting_heart_disease/test.csv";
    let sub_path = "examples/kaggle/predicting_heart_disease/submission.csv";

    if !Path::new(train_path).exists() {
        println!("Note: This script requires Kaggle S6E2 datasets.");
        return;
    }

    println!("1. Reading Data...");

    let mut train_rows: Vec<HeartDiseaseRow> = Vec::new();
    let mut train_rdr = csv::Reader::from_path(train_path).expect("Failed to open train.csv");
    for result in train_rdr.deserialize() {
        let row: HeartDiseaseRow = result.expect("Failed to deserialize train row");
        train_rows.push(row);
    }

    let mut test_rows: Vec<HeartDiseaseRow> = Vec::new();
    let mut test_rdr = csv::Reader::from_path(test_path).expect("Failed to open test.csv");
    for result in test_rdr.deserialize() {
        let row: HeartDiseaseRow = result.expect("Failed to deserialize test row");
        test_rows.push(row);
    }

    // Build vocabularies
    let mut sex_map = HashMap::new();
    let mut cp_map = HashMap::new();
    let mut fbs_map = HashMap::new();
    let mut ekg_map = HashMap::new();
    let mut angina_map = HashMap::new();
    let mut slope_map = HashMap::new();
    let mut thal_map = HashMap::new();

    let encode = |val: &str, map: &mut HashMap<String, f64>| -> f64 {
        let next_id = map.len() as f64;
        *map.entry(val.to_string()).or_insert(next_id)
    };

    println!("2. Advanced Feature Engineering...");

    let process_row = |row: &HeartDiseaseRow,
                       is_train: bool,
                       sex_map: &mut HashMap<String, f64>,
                       cp_map: &mut HashMap<String, f64>,
                       fbs_map: &mut HashMap<String, f64>,
                       ekg_map: &mut HashMap<String, f64>,
                       angina_map: &mut HashMap<String, f64>,
                       slope_map: &mut HashMap<String, f64>,
                       thal_map: &mut HashMap<String, f64>|
     -> (Vec<f64>, Option<f64>) {
        let mut feats = Vec::new();

        let sex = encode(&row.sex, sex_map);
        let cp = encode(&row.chest_pain_type, cp_map);
        let angina = encode(&row.exercise_angina, angina_map);
        let thal = encode(&row.thallium, thal_map);
        let slope = encode(&row.slope_of_st, slope_map);

        // Base features (13)
        feats.push(row.age); // 0
        feats.push(sex); // 1
        feats.push(cp); // 2
        feats.push(row.bp); // 3
        feats.push(row.cholesterol); // 4
        feats.push(encode(&row.fbs_over_120, fbs_map)); // 5
        feats.push(encode(&row.ekg_results, ekg_map)); // 6
        feats.push(row.max_hr); // 7
        feats.push(angina); // 8
        feats.push(row.st_depression); // 9
        feats.push(slope); // 10
        feats.push(row.vessels); // 11
        feats.push(thal); // 12

        // Advanced features
        // a. Risk Flags
        let high_bp = if row.bp > 140.0 { 1.0 } else { 0.0 };
        let high_chol = if row.cholesterol > 240.0 { 1.0 } else { 0.0 };
        let has_vessels = if row.vessels > 0.0 { 1.0 } else { 0.0 };
        feats.push(high_bp); // 13
        feats.push(high_chol); // 14
        feats.push(has_vessels); // 15

        // b. Risk Burden
        let risk_count = high_bp + high_chol + has_vessels;
        feats.push(risk_count); // 16
        feats.push(row.age * risk_count); // 17 age_risk interaction

        // c. Non-Linear Transforms
        feats.push((row.bp + 1.0).ln()); // 18
        feats.push((row.cholesterol + 1.0).ln()); // 19
        feats.push(row.age * row.age); // 20
        feats.push(row.max_hr.sqrt()); // 21

        // d. Domain Interactions (Categories)
        feats.push(sex * 10.0 + cp); // 22 sex_cp
        feats.push(thal * 10.0 + cp); // 23 thal_cp
        feats.push(thal * 10.0 + angina); // 24 thal_angina
        feats.push(slope * 10.0 + row.vessels); // 25 slope_vessels

        // e. Medical Ratios
        let mhr_pred = 220.0 - row.age;
        feats.push(if mhr_pred > 0.0 { row.max_hr / mhr_pred } else { 0.0 }); // 26 hr_age_ratio

        let target = if is_train {
            let t = row.heart_disease.as_ref().unwrap().to_lowercase();
            if t == "presence" || t == "1" {
                Some(1.0)
            } else {
                Some(0.0)
            }
        } else {
            None
        };
        (feats, target)
    };

    // Warm up map vocabularies
    for row in train_rows.iter() {
        process_row(
            row,
            true,
            &mut sex_map,
            &mut cp_map,
            &mut fbs_map,
            &mut ekg_map,
            &mut angina_map,
            &mut slope_map,
            &mut thal_map,
        );
    }
    for row in test_rows.iter() {
        process_row(
            row,
            false,
            &mut sex_map,
            &mut cp_map,
            &mut fbs_map,
            &mut ekg_map,
            &mut angina_map,
            &mut slope_map,
            &mut thal_map,
        );
    }

    let num_features = 27;
    let mut train_cols = vec![Vec::new(); num_features];
    let mut train_targets = Vec::new();

    for row in train_rows.iter() {
        let (feats, target) = process_row(
            row,
            true,
            &mut sex_map,
            &mut cp_map,
            &mut fbs_map,
            &mut ekg_map,
            &mut angina_map,
            &mut slope_map,
            &mut thal_map,
        );
        for (i, f) in feats.into_iter().enumerate() {
            train_cols[i].push(f);
        }
        train_targets.push(target.unwrap());
    }

    // Categorical feature indices (original + calculated interactions)
    let mut cat_feats = HashSet::new();
    cat_feats.insert(1);
    cat_feats.insert(2);
    cat_feats.insert(5);
    cat_feats.insert(6);
    cat_feats.insert(8);
    cat_feats.insert(10);
    cat_feats.insert(12);
    cat_feats.insert(22);
    cat_feats.insert(23);
    cat_feats.insert(24);
    cat_feats.insert(25);

    // Shuffle and Split
    let mut indices: Vec<usize> = (0..train_targets.len()).collect();
    let mut rng = StdRng::seed_from_u64(42);
    indices.shuffle(&mut rng);

    let n = train_targets.len();
    let split_idx = (n as f64 * 0.8) as usize;

    let mut cols_train = vec![vec![0.0; split_idx]; num_features];
    let mut y_train = Vec::with_capacity(split_idx);
    let mut cols_val = vec![vec![0.0; n - split_idx]; num_features];
    let mut y_val = Vec::with_capacity(n - split_idx);

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        if new_idx < split_idx {
            y_train.push(train_targets[old_idx]);
            for j in 0..num_features {
                cols_train[j][new_idx] = train_cols[j][old_idx];
            }
        } else {
            y_val.push(train_targets[old_idx]);
            for j in 0..num_features {
                cols_val[j][new_idx - split_idx] = train_cols[j][old_idx];
            }
        }
    }

    let matrix_train_refs: Vec<&[f64]> = cols_train.iter().map(|v| v.as_slice()).collect();
    let matrix_train = ColumnarMatrix::new(matrix_train_refs, None, split_idx);
    let matrix_val_refs: Vec<&[f64]> = cols_val.iter().map(|v| v.as_slice()).collect();
    let matrix_val = ColumnarMatrix::new(matrix_val_refs, None, n - split_idx);

    println!("3. Optimization (Search Budgets and Boldness)...");
    let budgets = vec![0.1, 0.2];
    let boldness_factors = vec![0.9, 1.0, 1.1];
    let mut best_ll = f64::MAX;
    let mut best_b = 1.0;
    let mut best_bf = 1.0;

    for &budget in &budgets {
        println!("Budget: {}", budget);
        let mut model = PerpetualBooster::default()
            .set_objective(Objective::LogLoss)
            .set_budget(budget)
            .set_categorical_features(Some(cat_feats.clone()))
            .set_iteration_limit(Some(1000))
            .set_log_iterations(1);

        model
            .fit_columnar(&matrix_train, &y_train, None, None)
            .expect("Fit failed");
        let alphas = vec![0.1];
        let _ = model.calibrate_columnar(CalibrationMethod::GRP, (&matrix_val, &y_val, &alphas));
        let raw_preds = model.predict_proba_columnar(&matrix_val, true, true);

        for &bf in &boldness_factors {
            let mut ls = 0.0;
            for i in 0..raw_preds.len() {
                let p = raw_preds[i].clamp(1e-7, 1.0 - 1e-7);
                let logit = (p / (1.0 - p)).ln() * bf;
                let prob = 1.0 / (1.0 + (-logit).exp());
                ls -= y_val[i] * prob.ln() + (1.0 - y_val[i]) * (1.0 - prob).ln();
            }
            let avg = ls / (raw_preds.len() as f64);
            println!("  B={:.1} bf={:.1} LL={:.4}", budget, bf, avg);
            if avg < best_ll {
                best_ll = avg;
                best_b = budget;
                best_bf = bf;
            }
        }
    }

    println!("Best: B={:.1} bf={:.1} LL={:.4}", best_b, best_bf, best_ll);

    println!("4. Final Training on Full Data...");
    let col_refs_all: Vec<&[f64]> = train_cols.iter().map(|v| v.as_slice()).collect();
    let matrix_all = ColumnarMatrix::new(col_refs_all, None, n);
    let mut final_model = PerpetualBooster::default()
        .set_objective(Objective::LogLoss)
        .set_budget(best_b)
        .set_categorical_features(Some(cat_feats));

    final_model
        .fit_columnar(&matrix_all, &train_targets, None, None)
        .expect("Final fit failed");
    let alphas_all = vec![0.1];
    let _ = final_model.calibrate_columnar(CalibrationMethod::GRP, (&matrix_all, &train_targets, &alphas_all));

    println!("5. Submission...");
    let mut test_cols = vec![Vec::new(); num_features];
    let mut test_ids = Vec::new();
    for row in test_rows.into_iter() {
        test_ids.push(row.id);
        let (feats, _) = process_row(
            &row,
            false,
            &mut sex_map,
            &mut cp_map,
            &mut fbs_map,
            &mut ekg_map,
            &mut angina_map,
            &mut slope_map,
            &mut thal_map,
        );
        for (i, f) in feats.into_iter().enumerate() {
            test_cols[i].push(f);
        }
    }

    let matrix_test_refs: Vec<&[f64]> = test_cols.iter().map(|v| v.as_slice()).collect();
    let matrix_test = ColumnarMatrix::new(matrix_test_refs, None, test_ids.len());
    let test_preds = final_model.predict_proba_columnar(&matrix_test, true, true);

    let mut file = File::create(sub_path).expect("Sub failed");
    writeln!(&mut file, "id,Heart Disease").unwrap();
    for (i, &id) in test_ids.iter().enumerate() {
        let p = test_preds[i].clamp(1e-7, 1.0 - 1e-7);
        let prob = 1.0 / (1.0 + (-((p / (1.0 - p)).ln() * best_bf)).exp());
        writeln!(&mut file, "{},{:.6}", id, prob).unwrap();
    }
    println!("Done. Winners incoming!");
}
