#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "library/perpetual.h"


SEXP R_wrapper_univariate(SEXP n_samplesSEXP, SEXP n_featuresSEXP) {
    
    if (!Rf_isInteger(n_samplesSEXP) || LENGTH(n_samplesSEXP) != 1)
        Rf_error("`n_samples` must be an integer scalar");
    if (!Rf_isInteger(n_featuresSEXP) || LENGTH(n_featuresSEXP) != 1)
        Rf_error("`n_features` must be an integer scalar");

    int n_samples  = INTEGER(n_samplesSEXP)[0];
    int n_features = INTEGER(n_featuresSEXP)[0];

    
    wrapper_univariate((uintptr_t)n_samples, (uintptr_t)n_features);

    
    return R_NilValue;
}

static const R_CallMethodDef CallEntries[] = {
    {"R_wrapper_univariate", (DL_FUNC) &R_wrapper_univariate, 2},
    {NULL, NULL, 0}
};


void R_init_perpetual(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
