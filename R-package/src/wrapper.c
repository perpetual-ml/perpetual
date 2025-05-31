#include <R.h>
#include <Rinternals.h>
#include <stdint.h>
#include "library/perpetual.h"



// ----------------------------------------------------------------------------
// 2.1 R wrapper for `train_univariate`
//   - Accepts two integer SEXP arguments: n_samples, n_features
//   - Calls train_univariate(...) to get PerpetualBoosterOpaque *
//   - Wraps the pointer in an R external pointer and returns it
// ----------------------------------------------------------------------------
SEXP R_train_univariate(SEXP s_n_samples, SEXP s_n_features) {
    // 2.1.1 Ensure the arguments are scalar integers
    if (!Rf_isInteger(s_n_samples) || LENGTH(s_n_samples) != 1) {
        Rf_error("`n_samples` must be a single integer");
    }
    if (!Rf_isInteger(s_n_features) || LENGTH(s_n_features) != 1) {
        Rf_error("`n_features` must be a single integer");
    }

    // 2.1.2 Convert R integers to uintptr_t
    uintptr_t n_samples = (uintptr_t)INTEGER(s_n_samples)[0];
    uintptr_t n_features = (uintptr_t)INTEGER(s_n_features)[0];

    // 2.1.3 Call the Rust/C function to train the model
    PerpetualBoosterOpaque *model_ptr = train_univariate(n_samples, n_features);
    if (model_ptr == NULL) {
        Rf_error("train_univariate returned NULL (training failed?)");
    }

    // 2.1.4 Wrap model_ptr in an R external pointer
    SEXP extptr = PROTECT(R_MakeExternalPtr((void *)model_ptr,
                                            R_NilValue,  // tag (unused)
                                            R_NilValue)); // protected object (none)
    // 2.1.5 Optionally, register a finalizer so that if R garbage collects this EXTPTR
    // before the user explicitly calls free, the model is freed automatically.
    // For simplicity, we skip an R-level finalizer here and let the user call `R_free_perpetual_booster`.

    // 2.1.6 Set a class attribute so R users see it as “PerpetualBooster”
    SEXP class_name = PROTECT(Rf_mkString("PerpetualBooster"));
    Rf_setAttrib(extptr, R_ClassSymbol, class_name);

    UNPROTECT(2); // extptr and class_name
    return extptr;
}

// ----------------------------------------------------------------------------
// 2.2 R wrapper for `predict_univariate`
//   - Accepts an external pointer to PerpetualBoosterOpaque (SEXP), plus two ints
//   - Extracts the pointer, calls predict_univariate(...), gets double* of length n_samples
//   - Copies that double* into an R numeric vector, calls free_predictions(...), then returns the vector
// ----------------------------------------------------------------------------
SEXP R_predict_univariate(SEXP s_model_ptr, SEXP s_n_samples, SEXP s_n_features) {

    // 2.2.2 Extract the raw pointer
    PerpetualBoosterOpaque *model_ptr = (PerpetualBoosterOpaque *)R_ExternalPtrAddr(s_model_ptr);
    if (model_ptr == NULL) {
        Rf_error("PerpetualBooster external pointer is NULL");
    }

    // 2.2.3 Check and convert n_samples and n_features
    if (!Rf_isInteger(s_n_samples) || LENGTH(s_n_samples) != 1) {
        Rf_error("`n_samples` must be a single integer");
    }
    if (!Rf_isInteger(s_n_features) || LENGTH(s_n_features) != 1) {
        Rf_error("`n_features` must be a single integer");
    }
    uintptr_t n_samples = (uintptr_t)INTEGER(s_n_samples)[0];
    uintptr_t n_features = (uintptr_t)INTEGER(s_n_features)[0];

    // 2.2.4 Call the Rust/C predict function
    double *c_preds = predict_univariate(model_ptr, n_samples, n_features);
    if (c_preds == NULL) {
        Rf_error("predict_univariate returned NULL (prediction failed?)");
    }

    // 2.2.5 Allocate an R numeric vector of length n_samples
    SEXP r_preds = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n_samples));
    double *rdata = REAL(r_preds);

    // 2.2.6 Copy from c_preds into rdata
    for (uintptr_t i = 0; i < n_samples; i++) {
        rdata[i] = c_preds[i];
    }

    // 2.2.7 Free the C-side prediction buffer
    free_predictions(c_preds, n_samples);

    UNPROTECT(1); // r_preds
    return r_preds;
}

// ----------------------------------------------------------------------------
// 2.3 R wrapper for `free_perpetual_booster`
//   - Accepts an external pointer; calls free_perpetual_booster(...) and then clears the external pointer
// ----------------------------------------------------------------------------
SEXP R_free_perpetual_booster(SEXP s_model_ptr) {

    PerpetualBoosterOpaque *model_ptr = (PerpetualBoosterOpaque *)R_ExternalPtrAddr(s_model_ptr);
    if (model_ptr != NULL) {
        free_perpetual_booster(model_ptr);
        // Clear the pointer in the external pointer so it cannot be used again
        R_ClearExternalPtr(s_model_ptr);
    }
    return R_NilValue; // .Call returns NULL in R
}

// ----------------------------------------------------------------------------
// 2.4 Registration of routines
//   We create an R_CallMethodDef array to register our .Call entry points.
// ----------------------------------------------------------------------------
static const R_CallMethodDef CallEntries[] = {
    {"R_train_univariate",   (DL_FUNC)&R_train_univariate,   2}, // two args: (int, int)
    {"R_predict_univariate", (DL_FUNC)&R_predict_univariate, 3}, // three args: (extptr, int, int)
    {"R_free_perpetual_booster", (DL_FUNC)&R_free_perpetual_booster, 1}, // one arg: (extptr)
    {NULL, NULL, 0}
};

// ----------------------------------------------------------------------------
// 2.5 Initialization function called when the shared object is loaded
//     R will automatically call this (if you link with `useDynLib(perpetual, .registration = TRUE)` in NAMESPACE).
// ----------------------------------------------------------------------------
void R_init_perpetual(DllInfo *dll) {
    // Register the .Call routines
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}