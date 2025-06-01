use std::sync::Arc;
use once_cell::sync::OnceCell;
use perpetual::objective_functions::{ObjectiveFunction};

// callbacks for custom objective
// functions
unsafe extern "C" {
    pub fn loss_callback(
        y: *const f64,
        yhat: *const f64,
        w: *const f64,
        n: usize,
        out: *mut f32,
    );

    pub fn gradient_callback(
        y: *const f64,
        yhat: *const f64,
        w: *const f64,
        n: usize,
        grad_out: *mut f32,
        hess_out: *mut f32,
    );

    pub fn initial_callback(
        y: *const f64,
        n: usize,
        w: *const f64,
    ) -> f64;

}

// 
#[derive(Clone)]
struct CObjective;

impl ObjectiveFunction for CObjective {
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>) -> Vec<f32> {
        let n = y.len();
        let mut out = Vec::<f32>::with_capacity(n);
        unsafe {
            out.set_len(n);
            loss_callback(
                y.as_ptr(),
                yhat.as_ptr(),
                sample_weight.map_or(std::ptr::null(), |w| w.as_ptr()),
                n,
                out.as_mut_ptr(),
            );
        }
        out
    }

    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let n = y.len();
        let mut grad = Vec::<f32>::with_capacity(n);
        let mut hess = Vec::<f32>::with_capacity(n);
        unsafe {
            grad.set_len(n);
            hess.set_len(n);
            gradient_callback(
                y.as_ptr(),
                yhat.as_ptr(),
                sample_weight.map_or(std::ptr::null(), |w| w.as_ptr()),
                n,
                grad.as_mut_ptr(),
                hess.as_mut_ptr(),
            );
        }
        (grad, Some(hess))
    }

    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>) -> f64 {
        unsafe {
            initial_callback(
                y.as_ptr(),
                y.len(),
                sample_weight.map_or(std::ptr::null(), |w| w.as_ptr()),
            )
        }
    }

    fn default_metric(&self) -> perpetual::metrics::Metric {
        perpetual::metrics::Metric::RootMeanSquaredError
    }
}


static GLOBAL_OBJECTIVE: OnceCell<Arc<dyn ObjectiveFunction>> = OnceCell::new();

pub fn get_global_objective() -> Option<Arc<dyn ObjectiveFunction>> {
    GLOBAL_OBJECTIVE.get().cloned()
}

#[unsafe(no_mangle)]
pub extern "C" fn register_custom_objective() {
    // Implementation for Rust side registration
    let _ = GLOBAL_OBJECTIVE.set(Arc::new(CObjective));
}
