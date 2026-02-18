use perpetual_rs::Matrix;
use perpetual_rs::MultiOutputBooster;
use perpetual_rs::PerpetualBooster as CratePerpetualBooster;
use perpetual_rs::booster::config::{
    BoosterConfig, BoosterIO, CalibrationMethod, ContributionsMethod, ImportanceMethod, MissingNodeTreatment,
};
use perpetual_rs::objective::Objective;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::slice;

// --- R API Definitions (Hand-Rolled) ---

type SEXP = *mut c_void;
#[allow(non_camel_case_types)]
type R_xlen_t = isize;

// SEXP Types
const REALSXP: u32 = 14;
const INTSXP: u32 = 13;
const STRSXP: u32 = 16;
const VECSXP: u32 = 19;

extern "C" {
    // Globals
    static R_NilValue: SEXP;
    static R_NamesSymbol: SEXP;

    // Memory Management
    fn Rf_protect(p: SEXP) -> SEXP;
    fn Rf_unprotect(n: c_int);

    // Allocation
    fn Rf_allocVector(type_: u32, length: R_xlen_t) -> SEXP;
    fn Rf_mkString(s: *const c_char) -> SEXP;
    fn Rf_mkChar(s: *const c_char) -> SEXP;

    // Accessors
    fn REAL(x: SEXP) -> *mut c_double;
    fn INTEGER(x: SEXP) -> *mut c_int;
    fn LOGICAL(x: SEXP) -> *mut c_int;
    fn STRING_ELT(x: SEXP, i: R_xlen_t) -> SEXP;
    fn SET_STRING_ELT(x: SEXP, i: R_xlen_t, v: SEXP);
    fn SET_VECTOR_ELT(x: SEXP, i: R_xlen_t, v: SEXP);
    fn R_CHAR(x: SEXP) -> *const c_char;
    fn Rf_xlength(x: SEXP) -> R_xlen_t;

    // Attributes
    fn Rf_setAttrib(x: SEXP, symbol: SEXP, val: SEXP);

    // External Pointers
    fn R_MakeExternalPtr(p: *mut c_void, tag: SEXP, prot: SEXP) -> SEXP;
    fn R_ExternalPtrAddr(p: SEXP) -> *mut c_void;
    fn R_RegisterCFinalizerEx(p: SEXP, fun: Option<unsafe extern "C" fn(SEXP)>, onexit: c_int);
    fn R_ClearExternalPtr(p: SEXP);

    // Printing
    fn Rprintf(format: *const c_char, ...);

    // Error Handling
    fn Rf_error(msg: *const c_char, ...) -> !;
}

// Helpers
unsafe fn r_error(msg: &str) -> ! {
    let s = CString::new(msg).unwrap_or(CString::new("Unknown error").unwrap());
    Rf_error(s.as_ptr());
}

macro_rules! r_safe {
    ($body:expr) => {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(result) => result,
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    *s
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.as_str()
                } else {
                    "Unknown panic in Rust code"
                };
                unsafe { r_error(&format!("Rust panic: {}", msg)) }
            }
        }
    };
}

trait OrRError<T> {
    fn or_r_error(self, msg: &str) -> T;
}

impl<T, E: std::fmt::Display> OrRError<T> for Result<T, E> {
    fn or_r_error(self, msg: &str) -> T {
        match self {
            Ok(v) => v,
            Err(e) => unsafe { r_error(&format!("{}: {}", msg, e)) },
        }
    }
}

impl<T> OrRError<T> for Option<T> {
    fn or_r_error(self, msg: &str) -> T {
        match self {
            Some(v) => v,
            None => unsafe { r_error(msg) },
        }
    }
}

unsafe fn is_nil(sexp: SEXP) -> bool {
    sexp == R_NilValue
}

unsafe fn get_f64(sexp: SEXP) -> f64 {
    *REAL(sexp)
}

unsafe fn get_int(sexp: SEXP) -> i32 {
    *INTEGER(sexp)
}

unsafe fn get_bool(sexp: SEXP) -> bool {
    let val = *LOGICAL(sexp);
    val != 0
}

unsafe fn get_str(sexp: SEXP) -> String {
    let char_ptr = R_CHAR(STRING_ELT(sexp, 0));
    CStr::from_ptr(char_ptr).to_string_lossy().into_owned()
}

// Helpers for Options
unsafe fn opt_f64(sexp: SEXP) -> Option<f64> {
    if is_nil(sexp) { None } else { Some(get_f64(sexp)) }
}

unsafe fn opt_i32(sexp: SEXP) -> Option<i32> {
    if is_nil(sexp) { None } else { Some(get_int(sexp)) }
}

unsafe fn opt_bool(sexp: SEXP) -> Option<bool> {
    if is_nil(sexp) { None } else { Some(get_bool(sexp)) }
}

unsafe fn opt_str(sexp: SEXP) -> Option<String> {
    if is_nil(sexp) { None } else { Some(get_str(sexp)) }
}

// --- Logic ---

enum InternalBooster {
    Single(CratePerpetualBooster),
    Multi(MultiOutputBooster),
}

pub struct PerpetualBoosterWrapper {
    internal: Option<InternalBooster>,
    config: BoosterConfig,
    classes: Vec<f64>,
}

// Finalizer for the external pointer
pub unsafe extern "C" fn finalizer(ptr: SEXP) {
    if R_ExternalPtrAddr(ptr).is_null() {
        return;
    }
    let _ = Box::from_raw(R_ExternalPtrAddr(ptr) as *mut PerpetualBoosterWrapper);
    R_ClearExternalPtr(ptr);
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_new(
    objective: SEXP,
    budget: SEXP,
    max_bin: SEXP,
    num_threads: SEXP,
    missing: SEXP,
    allow_missing_splits: SEXP,
    create_missing_branch: SEXP,
    missing_node_treatment: SEXP,
    log_iterations: SEXP,
    quantile: SEXP,
    reset: SEXP,
    timeout: SEXP,
    iteration_limit: SEXP,
    memory_limit: SEXP,
    stopping_rounds: SEXP,
    seed: SEXP,
    calibration_method: SEXP,
    save_node_stats: SEXP,
) -> SEXP {
    r_safe!(unsafe {
        let mut config = BoosterConfig::default();

        if let Some(obj) = opt_str(objective) {
            if let Ok(obj_enum) = serde_plain::from_str::<Objective>(&obj) {
                config.objective = obj_enum;
            } else if !obj.is_empty() {
                let msg = CString::new(format!(
                    "Warning: Unknown objective '{}'. Using default LogLoss.\n",
                    obj
                ))
                .unwrap();
                Rprintf(msg.as_ptr());
            }
        }

        if let Some(b) = opt_f64(budget) {
            config.budget = b as f32;
        }
        if let Some(mb) = opt_i32(max_bin) {
            config.max_bin = mb as u16;
        }
        if let Some(nt) = opt_i32(num_threads) {
            if nt > 0 {
                config.num_threads = Some(nt as usize);
            }
        }
        if let Some(val) = opt_f64(missing) {
            config.missing = val;
        }
        if let Some(val) = opt_bool(allow_missing_splits) {
            config.allow_missing_splits = val;
        }
        if let Some(val) = opt_bool(create_missing_branch) {
            config.create_missing_branch = val;
        }

        if let Some(treatment) = opt_str(missing_node_treatment) {
            if let Ok(t) = serde_plain::from_str::<MissingNodeTreatment>(&treatment) {
                config.missing_node_treatment = t;
            }
        }

        if let Some(m_str) = opt_str(calibration_method) {
            if let Ok(m) = serde_plain::from_str::<CalibrationMethod>(&m_str) {
                config.calibration_method = m;
            }
        }

        if let Some(val) = opt_i32(log_iterations) {
            config.log_iterations = val as usize;
        }
        if let Some(val) = opt_f64(quantile) {
            config.quantile = Some(val);
        }
        if let Some(val) = opt_bool(reset) {
            config.reset = Some(val);
        }
        if let Some(val) = opt_f64(timeout) {
            config.timeout = Some(val as f32);
        }
        if let Some(val) = opt_i32(iteration_limit) {
            config.iteration_limit = Some(val as usize);
        }
        if let Some(val) = opt_f64(memory_limit) {
            config.memory_limit = Some(val as f32);
        }
        if let Some(val) = opt_i32(stopping_rounds) {
            config.stopping_rounds = Some(val as usize);
        }
        if let Some(val) = opt_i32(seed) {
            config.seed = val as u64;
        }

        if let Some(val) = opt_bool(save_node_stats) {
            config.save_node_stats = val;
        }

        let booster = PerpetualBoosterWrapper {
            internal: None,
            config,
            classes: Vec::new(),
        };

        let ptr = R_MakeExternalPtr(Box::into_raw(Box::new(booster)) as *mut _, R_NilValue, R_NilValue);
        R_RegisterCFinalizerEx(ptr, Some(finalizer), 1);
        ptr
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_fit(ptr: SEXP, flat_data: SEXP, rows: SEXP, cols: SEXP, y: SEXP) -> SEXP {
    r_safe!(unsafe {
        let booster = &mut *(R_ExternalPtrAddr(ptr) as *mut PerpetualBoosterWrapper);

        let rows_val = get_int(rows) as usize;
        let cols_val = get_int(cols) as usize;

        let data_slice = slice::from_raw_parts(REAL(flat_data), rows_val * cols_val);
        let y_slice = slice::from_raw_parts(REAL(y), Rf_xlength(y) as usize);

        let _data_vec = data_slice.to_vec();
        let matrix = Matrix::new(&_data_vec, rows_val, cols_val);

        // Detect unique classes
        let mut unique_classes: Vec<f64> = y_slice.iter().filter(|v| !v.is_nan()).cloned().collect();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_classes.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
        booster.classes = unique_classes;

        if booster.classes.len() > 2 && matches!(booster.config.objective, Objective::LogLoss) {
            // Multiclass logic
            let n_classes = booster.classes.len();
            let mut y_matrix_data = vec![0.0; rows_val * n_classes];
            for (i, &val) in y_slice.iter().enumerate() {
                if let Some(pos) = booster.classes.iter().position(|&c| (c - val).abs() < f64::EPSILON) {
                    y_matrix_data[pos * rows_val + i] = 1.0;
                }
            }
            let y_matrix = Matrix::new(&y_matrix_data, rows_val, n_classes);

            let mut multi_booster = MultiOutputBooster::new(
                n_classes,
                booster.config.objective.clone(),
                booster.config.budget,
                booster.config.max_bin,
                booster.config.num_threads,
                booster.config.monotone_constraints.clone(),
                booster.config.interaction_constraints.clone(),
                booster.config.force_children_to_bound_parent,
                booster.config.missing,
                booster.config.allow_missing_splits,
                booster.config.create_missing_branch,
                booster.config.terminate_missing_features.clone(),
                booster.config.missing_node_treatment,
                booster.config.log_iterations,
                booster.config.seed,
                booster.config.quantile,
                booster.config.reset,
                booster.config.categorical_features.clone(),
                booster.config.timeout,
                booster.config.iteration_limit,
                booster.config.memory_limit,
                booster.config.stopping_rounds,
                booster.config.save_node_stats,
                booster.config.calibration_method.clone(),
            )
            .or_r_error("Failed to create MultiOutputBooster");

            multi_booster
                .fit(&matrix, &y_matrix, None, None)
                .or_r_error("Failed to fit MultiOutputBooster");

            let classes_str = booster
                .classes
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",");
            for b_ in &mut multi_booster.boosters {
                b_.metadata.insert("classes".to_string(), classes_str.clone());
            }
            booster.internal = Some(InternalBooster::Multi(multi_booster));
        } else {
            // Single booster
            let mut single_booster = CratePerpetualBooster::new(
                booster.config.objective.clone(),
                booster.config.budget,
                f64::NAN,
                booster.config.max_bin,
                booster.config.num_threads,
                booster.config.monotone_constraints.clone(),
                booster.config.interaction_constraints.clone(),
                booster.config.force_children_to_bound_parent,
                booster.config.missing,
                booster.config.allow_missing_splits,
                booster.config.create_missing_branch,
                booster.config.terminate_missing_features.clone(),
                booster.config.missing_node_treatment,
                booster.config.log_iterations,
                booster.config.seed,
                booster.config.quantile,
                booster.config.reset,
                booster.config.categorical_features.clone(),
                booster.config.timeout,
                booster.config.iteration_limit,
                booster.config.memory_limit,
                booster.config.stopping_rounds,
                booster.config.save_node_stats,
                booster.config.calibration_method.clone(),
            )
            .or_r_error("Failed to create PerpetualBooster");

            single_booster
                .fit(&matrix, &y_slice.to_vec(), None, None::<&[u64]>)
                .or_r_error("Failed to fit PerpetualBooster");

            if !booster.classes.is_empty() {
                let classes_str = booster
                    .classes
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                single_booster.metadata.insert("classes".to_string(), classes_str);
            }
            booster.internal = Some(InternalBooster::Single(single_booster));
        }

        R_NilValue
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_predict(ptr: SEXP, flat_data: SEXP, rows: SEXP, cols: SEXP) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let rows_val = get_int(rows) as usize;
        let cols_val = get_int(cols) as usize;
        let data_slice = slice::from_raw_parts(REAL(flat_data), rows_val * cols_val);
        let _data_vec = data_slice.to_vec();
        let matrix = Matrix::new(&_data_vec, rows_val, cols_val);

        let preds = match &booster.internal {
            Some(InternalBooster::Single(b)) => b.predict(&matrix, true),
            Some(InternalBooster::Multi(b)) => b.predict(&matrix, true),
            None => r_error("Booster not fitted"),
        };

        let result = Rf_protect(Rf_allocVector(REALSXP, preds.len() as R_xlen_t));
        let r_data = REAL(result);
        for (i, val) in preds.iter().enumerate() {
            *r_data.add(i) = *val;
        }
        Rf_unprotect(1);
        result
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_predict_proba(ptr: SEXP, flat_data: SEXP, rows: SEXP, cols: SEXP) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let rows_val = get_int(rows) as usize;
        let cols_val = get_int(cols) as usize;
        let data_slice = slice::from_raw_parts(REAL(flat_data), rows_val * cols_val);
        let _data_vec = data_slice.to_vec();
        let matrix = Matrix::new(&_data_vec, rows_val, cols_val);

        let preds = match &booster.internal {
            Some(InternalBooster::Single(b)) => b.predict_proba(&matrix, true, false),
            Some(InternalBooster::Multi(b)) => b.predict_proba(&matrix, true),
            None => r_error("Booster not fitted"),
        };

        let result = Rf_protect(Rf_allocVector(REALSXP, preds.len() as R_xlen_t));
        let r_data = REAL(result);
        for (i, val) in preds.iter().enumerate() {
            *r_data.add(i) = *val;
        }
        Rf_unprotect(1);
        result
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_save_booster(ptr: SEXP, path: SEXP) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let path_str = get_str(path);
        match &booster.internal {
            Some(InternalBooster::Single(b)) => b.save_booster(&path_str).or_r_error("Failed to save single booster"),
            Some(InternalBooster::Multi(b)) => b.save_booster(&path_str).or_r_error("Failed to save multi booster"),
            None => r_error("Booster not fitted"),
        }
        R_NilValue
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_load_booster(path: SEXP) -> SEXP {
    r_safe!(unsafe {
        let path_str = get_str(path);
        let json_str = std::fs::read_to_string(path_str).or_r_error("Failed to read file");
        let mut classes = Vec::new();

        let internal = if json_str.contains("\"boosters\":") {
            let b = MultiOutputBooster::from_json(&json_str).or_r_error("Failed to parse MultiOutputBooster JSON");
            if let Some(metadata) = b.boosters.first().map(|b_| &b_.metadata) {
                if let Some(c_str) = metadata.get("classes") {
                    classes = c_str.split(',').filter_map(|s| s.parse::<f64>().ok()).collect();
                }
            }
            Some(InternalBooster::Multi(b))
        } else {
            let b = CratePerpetualBooster::from_json(&json_str).or_r_error("Failed to parse PerpetualBooster JSON");
            if let Some(c_str) = b.metadata.get("classes") {
                classes = c_str.split(',').filter_map(|s| s.parse::<f64>().ok()).collect();
            }
            Some(InternalBooster::Single(b))
        };

        let config = match internal.as_ref().or_r_error("Internal error: internal booster is None") {
            InternalBooster::Single(b) => b.cfg.clone(),
            InternalBooster::Multi(b) => b.cfg.clone(),
        };

        let booster = PerpetualBoosterWrapper {
            internal,
            config,
            classes,
        };

        let ptr = R_MakeExternalPtr(Box::into_raw(Box::new(booster)) as *mut _, R_NilValue, R_NilValue);
        R_RegisterCFinalizerEx(ptr, Some(finalizer), 1);
        ptr
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_json_dump(ptr: SEXP) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let dump = match &booster.internal {
            Some(InternalBooster::Single(b)) => b.json_dump().or_r_error("Failed to dump JSON"),
            Some(InternalBooster::Multi(b)) => b.json_dump().or_r_error("Failed to dump JSON"),
            None => r_error("Booster not fitted"),
        };

        let result = Rf_protect(Rf_mkString(
            CString::new(dump)
                .or_r_error("Failed to convert JSON to CString")
                .as_ptr(),
        ));
        Rf_unprotect(1);
        result
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_number_of_trees(ptr: SEXP) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let count = match &booster.internal {
            Some(InternalBooster::Single(b)) => b.get_prediction_trees().len() as i32,
            Some(InternalBooster::Multi(b)) => b
                .boosters
                .iter()
                .map(|b_| b_.get_prediction_trees().len())
                .sum::<usize>() as i32,
            None => 0,
        };
        let result = Rf_protect(Rf_allocVector(INTSXP, 1));
        *INTEGER(result) = count;
        Rf_unprotect(1);
        result
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_base_score(ptr: SEXP) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let score = match &booster.internal {
            Some(InternalBooster::Single(b)) => b.base_score,
            Some(InternalBooster::Multi(b)) => b.boosters.first().map(|b_| b_.base_score).unwrap_or(f64::NAN),
            None => f64::NAN,
        };
        let result = Rf_protect(Rf_allocVector(REALSXP, 1));
        *REAL(result) = score;
        Rf_unprotect(1);
        result
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_predict_contributions(
    ptr: SEXP,
    flat_data: SEXP,
    rows: SEXP,
    cols: SEXP,
    method: SEXP,
) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let rows_val = get_int(rows) as usize;
        let cols_val = get_int(cols) as usize;
        let data_slice = slice::from_raw_parts(REAL(flat_data), rows_val * cols_val);
        let _data_vec = data_slice.to_vec();
        let matrix = Matrix::new(&_data_vec, rows_val, cols_val);
        let method_str = get_str(method);

        let method_enum = match method_str.to_lowercase().as_str() {
            "weight" => ContributionsMethod::Weight,
            "average" => ContributionsMethod::Average,
            "branchdifference" | "branch_difference" => ContributionsMethod::BranchDifference,
            "midpointdifference" | "midpoint_difference" => ContributionsMethod::MidpointDifference,
            "modedifference" | "mode_difference" => ContributionsMethod::ModeDifference,
            "probabilitychange" | "probability_change" => ContributionsMethod::ProbabilityChange,
            "shapley" => ContributionsMethod::Shapley,
            _ => ContributionsMethod::Average,
        };

        let contribs = match &booster.internal {
            Some(InternalBooster::Single(b)) => b.predict_contributions(&matrix, method_enum, true),
            Some(InternalBooster::Multi(b)) => b.predict_contributions(&matrix, method_enum, true),
            None => r_error("Booster not fitted"),
        };

        let result = Rf_protect(Rf_allocVector(REALSXP, contribs.len() as R_xlen_t));
        let r_data = REAL(result);
        for (i, val) in contribs.iter().enumerate() {
            *r_data.add(i) = *val;
        }
        Rf_unprotect(1);
        result
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_calibrate(
    ptr: SEXP,
    flat_data: SEXP,
    rows: SEXP,
    cols: SEXP,
    y: SEXP,
    flat_data_cal: SEXP,
    rows_cal: SEXP,
    cols_cal: SEXP,
    y_cal: SEXP,
    alpha: SEXP,
    method: SEXP,
) -> SEXP {
    r_safe!(unsafe {
        let booster = &mut *(R_ExternalPtrAddr(ptr) as *mut PerpetualBoosterWrapper);

        if let Some(m_str) = opt_str(method) {
            if let Ok(m) = serde_plain::from_str::<CalibrationMethod>(&m_str) {
                booster.config.calibration_method = m;
            }
        }

        let rows_cal_val = get_int(rows_cal) as usize;
        let cols_cal_val = get_int(cols_cal) as usize;
        let data_cal_slice = slice::from_raw_parts(REAL(flat_data_cal), rows_cal_val * cols_cal_val);
        let y_cal_slice = slice::from_raw_parts(REAL(y_cal), Rf_xlength(y_cal) as usize);
        let alpha_slice = slice::from_raw_parts(REAL(alpha), Rf_xlength(alpha) as usize);

        let _data_cal_vec = data_cal_slice.to_vec();
        let matrix_cal = Matrix::new(&_data_cal_vec, rows_cal_val, cols_cal_val);
        let _y_cal_vec = y_cal_slice.to_vec();
        let _alpha_vec = alpha_slice.to_vec();

        match &mut booster.internal {
            Some(InternalBooster::Single(b)) => {
                let curr_method = booster.config.calibration_method;
                let is_classification = matches!(booster.config.objective, Objective::LogLoss);

                if !is_classification && matches!(curr_method, CalibrationMethod::Conformal) {
                    let rows_val = get_int(rows) as usize;
                    let cols_val = get_int(cols) as usize;
                    let data_slice = slice::from_raw_parts(REAL(flat_data), rows_val * cols_val);
                    let y_slice = slice::from_raw_parts(REAL(y), Rf_xlength(y) as usize);

                    let _data_vec = data_slice.to_vec();
                    let matrix = Matrix::new(&_data_vec, rows_val, cols_val);
                    let _y_vec = y_slice.to_vec();

                    b.calibrate_conformal(&matrix, &_y_vec, None, None, (&matrix_cal, &_y_cal_vec, &_alpha_vec))
                        .or_r_error("Failed to calibrate (conformal)");
                } else {
                    b.calibrate(curr_method, (&matrix_cal, &_y_cal_vec, &_alpha_vec))
                        .or_r_error("Failed to calibrate");
                }
            }
            _ => r_error("Calibration is only supported for single-output boosters currently"),
        }
        R_NilValue
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_predict_intervals(
    ptr: SEXP,
    flat_data: SEXP,
    rows: SEXP,
    cols: SEXP,
) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let rows_val = get_int(rows) as usize;
        let cols_val = get_int(cols) as usize;
        let data_slice = slice::from_raw_parts(REAL(flat_data), rows_val * cols_val);
        let _data_vec = data_slice.to_vec();
        let matrix = Matrix::new(&_data_vec, rows_val, cols_val);

        let result = match &booster.internal {
            Some(InternalBooster::Single(b)) => {
                let intervals = b.predict_intervals(&matrix, true);
                let r_list = Rf_protect(Rf_allocVector(VECSXP, intervals.len() as R_xlen_t));

                // Set names for the list (alphas)
                let names_vec = Rf_allocVector(STRSXP, intervals.len() as R_xlen_t);
                Rf_protect(names_vec);

                for (i, (alpha, preds)) in intervals.into_iter().enumerate() {
                    // Set name
                    let s = CString::new(alpha.to_string()).unwrap();
                    SET_STRING_ELT(names_vec, i as R_xlen_t, Rf_mkChar(s.as_ptr()));

                    // Set content
                    let total_len: usize = preds.iter().map(|v| v.len()).sum();
                    let alpha_vec = Rf_allocVector(REALSXP, total_len as R_xlen_t);
                    let r_alpha = REAL(alpha_vec);
                    for (j, p) in preds.into_iter().flatten().enumerate() {
                        *r_alpha.add(j) = p;
                    }
                    SET_VECTOR_ELT(r_list, i as R_xlen_t, alpha_vec);
                }

                Rf_setAttrib(r_list, R_NamesSymbol, names_vec);
                Rf_unprotect(2); // names_vec and r_list (wait, r_list must be protected until return if referenced, but here we return it)
                // Actually Rf_unprotect pops items from stack.
                // Stack: [r_list, names_vec] (top)
                // Rf_unprotect(2) pops both.
                // But we need to return r_list.
                // If we unprotect, is it safe? Yes if we don't allocate more.
                // BUT usually we return a protected value or it's implicitly part of result logic?
                // In R extensions: "If you want to return a protected object, you should unprotect it first but better make sure it's safe."
                // Typically: protect result -> do work -> unprotect(n) -> return result.
                // If result is on stack, unprotecting it makes it vulnerable to GC if we allocate?
                // No, we are returning immediately. And R function result is protected by caller.
                // Rf_unprotect will happen at end
                r_list
            }
            _ => r_error("Prediction intervals not supported"),
        };

        // The previous code had Rf_unprotect(1) AFTER the match, assuming only r_list was protected.
        // Now I do cleanup INSIDE the match for both r_list and names_vec.
        // So I should remove the outer Rf_unprotect(1).

        result
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_calculate_feature_importance(
    ptr: SEXP,
    method: SEXP,
    normalize: SEXP,
) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let method_str = get_str(method);
        let normalize_val = get_bool(normalize);

        let method_enum = match method_str.to_lowercase().as_str() {
            "weight" => ImportanceMethod::Weight,
            "gain" => ImportanceMethod::Gain,
            "totalgain" | "total_gain" => ImportanceMethod::TotalGain,
            "cover" => ImportanceMethod::Cover,
            "totalcover" | "total_cover" => ImportanceMethod::TotalCover,
            _ => ImportanceMethod::Gain,
        };

        let importance = match &booster.internal {
            Some(InternalBooster::Single(b)) => b.calculate_feature_importance(method_enum, normalize_val),
            Some(InternalBooster::Multi(b)) => b.calculate_feature_importance(method_enum, normalize_val),
            None => r_error("Booster not fitted"),
        };

        let r_list = Rf_protect(Rf_allocVector(REALSXP, importance.len() as R_xlen_t));
        let r_vals = REAL(r_list);

        // We want a named vector
        let names_vec = Rf_allocVector(STRSXP, importance.len() as R_xlen_t);
        Rf_protect(names_vec);

        for (i, (idx, val)) in importance.into_iter().enumerate() {
            *r_vals.add(i) = val as f64;
            let s = CString::new(idx.to_string()).unwrap();
            SET_STRING_ELT(names_vec, i as R_xlen_t, Rf_mkChar(s.as_ptr()));
        }

        Rf_setAttrib(r_list, R_NamesSymbol, names_vec);
        Rf_unprotect(2); // names_vec and r_list

        r_list
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_get_classes(ptr: SEXP) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let result = Rf_protect(Rf_allocVector(REALSXP, booster.classes.len() as R_xlen_t));
        let r_data = REAL(result);
        for (i, val) in booster.classes.iter().enumerate() {
            *r_data.add(i) = *val;
        }
        Rf_unprotect(1);
        result
    })
}

#[no_mangle]
pub unsafe extern "C" fn PerpetualBooster_get_objective(ptr: SEXP) -> SEXP {
    r_safe!(unsafe {
        let booster = &*(R_ExternalPtrAddr(ptr) as *const PerpetualBoosterWrapper);
        let obj_str = match booster.config.objective {
            Objective::LogLoss => "LogLoss",
            Objective::SquaredLoss => "SquaredLoss",
            Objective::QuantileLoss { .. } => "QuantileLoss",
            Objective::HuberLoss { .. } => "HuberLoss",
            Objective::AdaptiveHuberLoss { .. } => "AdaptiveHuberLoss",
            Objective::ListNetLoss => "ListNetLoss",
            Objective::Custom(_) => "Custom",
        };
        let result = Rf_protect(Rf_mkString(CString::new(obj_str).unwrap().as_ptr()));
        Rf_unprotect(1);
        result
    })
}

#[no_mangle]
pub unsafe extern "C" fn test_binding() -> SEXP {
    r_safe!(unsafe {
        let msg = CString::new("Binding OK").unwrap();
        Rf_mkString(msg.as_ptr())
    })
}
