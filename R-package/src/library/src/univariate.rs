use perpetual::Matrix;
use perpetual::UnivariateBooster;
use perpetual::objective_functions::*;

// Opaque pointer
#[repr(C)]
pub struct OpaquePointer {
    _private: [u8; 0],
}

/// Engine
///
/// Builds a Perpetual::default() model and returns a
/// opaque pointer that that can be modified elsewhere.
///
/// ## TODO:
///
/// * Allow for passing objective functions
/// * Allow for passing custom objective functions
///
/// ## NOTE:
///
/// It currently only uses SquaredLoss
#[unsafe(no_mangle)]
pub extern "C" fn engine() -> *mut OpaquePointer {
    // initialize the model
    let mut model: UnivariateBooster = UnivariateBooster::default().set_objective(Objective::SquaredLoss);

    // 4.4 Box the model on the heap and cast to the opaque type
    let boxed_model = Box::new(model);
    let raw_ptr: *mut UnivariateBooster = Box::into_raw(boxed_model);
    raw_ptr as *mut OpaquePointer
}

/// Tune
///
/// Sets the tuning parameters of the booster.
///
/// ## Parameters
///
/// * `budget`: `f32` the buget parameter `set_budget()`
/// * `max_bin`: `u16` the maximum numner of bins `set_max_bin()`
///
/// ## TODO:
///
/// * Expand number of parameters passable
#[unsafe(no_mangle)]
pub extern "C" fn tune(model_ptr: *mut OpaquePointer, budget: f32, max_bin: u16) {
    // recast pointer
    // let booster_ptr = model_ptr as *mut UnivariateBooster;
    let model: &UnivariateBooster = unsafe { &*(model_ptr as *mut UnivariateBooster) };

    model.clone().set_budget(budget).set_max_bin(max_bin);

    // unsafe {
    //     if let Some(model) = booster_ptr.as_mut() {
    //         booster.clone().set_budget(budget).set_max_bin(max_bin);
    //     } else {
    //         eprintln!("Tune reciev null pointer")
    //     }
    // }
}

/// Train
///
/// ## Parameters
/// * `x_vector`: A `Vec<f64>` of flattened features
/// * `y_vector`: A `Vec<f64>` of flattened targets
/// * `w_vector`: A `Option<Vec<f64>>` of flattened sample weights
/// * `x_cols`: A usize corresponding the the number of features in preflattened X
///
#[unsafe(no_mangle)]
pub extern "C" fn train(
    model_ptr: *mut OpaquePointer,
    x_vector: Vec<f64>,
    y_vector: Vec<f64>,
    w_vector: Option<Vec<f64>>,
    x_cols: usize,
) -> *mut OpaquePointer {
    // TODO: Throw error message
    if model_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let booster_raw: *mut UnivariateBooster = model_ptr as *mut UnivariateBooster;

    unsafe {
        let booster_ref: &mut UnivariateBooster = &mut *booster_raw;

        let matrix = Matrix::new(&x_vector, x_vector.len(), x_cols);
        let w_slice: Option<&[f64]> = w_vector.as_ref().map(|v| v.as_slice());

        booster_ref.fit(&matrix, &y_vector, w_slice);
    }

    model_ptr
}

/// Predict
/// 
/// ## Parameters
/// * `x_vector`: A `Vec<f64>` of flattened features
/// * `w_vector`: A `Option<Vec<f64>>` of flattened sample weights
/// * `x_cols`: A usize corresponding the the number of features in preflattened X
/// 
/// ## NOTE:
/// 
/// It is probably a better idea to implement it as
/// `predict_numeric`, `predict_tree` etc.
/// - Or maybe an enum?
#[unsafe(no_mangle)]
pub extern "C" fn predict(
    model_ptr: *mut OpaquePointer,
    x_vector: Vec<f64>,
    x_cols: usize) -> *mut f64 {

    // TODO: Throw error message
    if model_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let model: &UnivariateBooster = unsafe { &*(model_ptr as *mut UnivariateBooster) };


    let matrix = Matrix::new(&x_vector, x_vector.len(), x_cols);

    let preds: Vec<f64> = model.predict(&matrix, false);

    let mut boxed_slice = preds.into_boxed_slice();
    let ptr_to_data = boxed_slice.as_mut_ptr();
    std::mem::forget(boxed_slice);

    ptr_to_data
}

// Clear memory
#[unsafe(no_mangle)]
pub extern "C" fn free_perpetual_booster(model_ptr: *mut OpaquePointer) {
    if model_ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(model_ptr as *mut UnivariateBooster);
    }
}


#[unsafe(no_mangle)]
pub extern "C" fn free_predictions(ptr: *mut f64, length: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let slice = std::slice::from_raw_parts_mut(ptr, length);
        Box::from_raw(slice as *mut [f64]);
    }
}
