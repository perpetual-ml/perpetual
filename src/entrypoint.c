#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// Declare the Rust functions
extern SEXP PerpetualBooster_new(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP PerpetualBooster_fit(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP PerpetualBooster_predict(SEXP, SEXP, SEXP, SEXP);
extern SEXP PerpetualBooster_predict_proba(SEXP, SEXP, SEXP, SEXP);
extern SEXP PerpetualBooster_save_booster(SEXP, SEXP);
extern SEXP PerpetualBooster_load_booster(SEXP);
extern SEXP PerpetualBooster_json_dump(SEXP);
extern SEXP PerpetualBooster_number_of_trees(SEXP);
extern SEXP PerpetualBooster_base_score(SEXP);
extern SEXP PerpetualBooster_predict_contributions(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP PerpetualBooster_calibrate(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP PerpetualBooster_predict_intervals(SEXP, SEXP, SEXP, SEXP);
extern SEXP PerpetualBooster_calculate_feature_importance(SEXP, SEXP, SEXP);
extern SEXP PerpetualBooster_get_classes(SEXP);
extern SEXP PerpetualBooster_get_objective(SEXP);
extern SEXP test_binding();

static const R_CallMethodDef CallEntries[] = {
    {"PerpetualBooster_new", (DL_FUNC) &PerpetualBooster_new, 18},
    {"PerpetualBooster_fit", (DL_FUNC) &PerpetualBooster_fit, 5},
    {"PerpetualBooster_predict", (DL_FUNC) &PerpetualBooster_predict, 4},
    {"PerpetualBooster_predict_proba", (DL_FUNC) &PerpetualBooster_predict_proba, 4},
    {"PerpetualBooster_save_booster", (DL_FUNC) &PerpetualBooster_save_booster, 2},
    {"PerpetualBooster_load_booster", (DL_FUNC) &PerpetualBooster_load_booster, 1},
    {"PerpetualBooster_json_dump", (DL_FUNC) &PerpetualBooster_json_dump, 1},
    {"PerpetualBooster_number_of_trees", (DL_FUNC) &PerpetualBooster_number_of_trees, 1},
    {"PerpetualBooster_base_score", (DL_FUNC) &PerpetualBooster_base_score, 1},
    {"PerpetualBooster_predict_contributions", (DL_FUNC) &PerpetualBooster_predict_contributions, 5},
    {"PerpetualBooster_calibrate", (DL_FUNC) &PerpetualBooster_calibrate, 11},
    {"PerpetualBooster_predict_intervals", (DL_FUNC) &PerpetualBooster_predict_intervals, 4},
    {"PerpetualBooster_calculate_feature_importance", (DL_FUNC) &PerpetualBooster_calculate_feature_importance, 3},
    {"PerpetualBooster_get_classes", (DL_FUNC) &PerpetualBooster_get_classes, 1},
    {"PerpetualBooster_get_objective", (DL_FUNC) &PerpetualBooster_get_objective, 1},
    {"test_binding", (DL_FUNC) &test_binding, 0},
    {NULL, NULL, 0}
};

void R_init_perpetual(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

// CRAN policy: Packages should not call abort().
// We override it to call Rf_error instead, which handles the error broadly within R.
// On Windows, visibility hidden is not supported/needed for this static linking case, and triggers a warning.
#ifdef _WIN32
void abort(void) {
#else
__attribute__((visibility("hidden"))) void abort(void) {
#endif
    Rf_error("Rust code attempted to abort (panic).");
}
