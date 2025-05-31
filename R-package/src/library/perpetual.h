#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Option_Vec_f64 Option_Vec_f64;

typedef struct Vec_f64 Vec_f64;

typedef struct OpaquePointer {
  uint8_t _private[0];
} OpaquePointer;

/**
 * Engine
 *
 * Builds a Perpetual::default() model and returns a
 * opaque pointer that that can be modified elsewhere.
 *
 * ## TODO:
 *
 * * Allow for passing objective functions
 * * Allow for passing custom objective functions
 *
 * ## NOTE:
 *
 * It currently only uses SquaredLoss
 */
struct OpaquePointer *engine(void);

/**
 * Tune
 *
 * Sets the tuning parameters of the booster.
 *
 * ## Parameters
 *
 * * `budget`: `f32` the buget parameter `set_budget()`
 * * `max_bin`: `u16` the maximum numner of bins `set_max_bin()`
 *
 * ## TODO:
 *
 * * Expand number of parameters passable
 */
void tune(struct OpaquePointer *model_ptr, float budget, uint16_t max_bin);

/**
 * Train
 *
 * ## Parameters
 * * `x_vector`: A `Vec<f64>` of flattened features
 * * `y_vector`: A `Vec<f64>` of flattened targets
 * * `w_vector`: A `Option<Vec<f64>>` of flattened sample weights
 * * `x_cols`: A usize corresponding the the number of features in preflattened X
 *
 */
struct OpaquePointer *train(struct OpaquePointer *model_ptr,
                            struct Vec_f64 x_vector,
                            struct Vec_f64 y_vector,
                            struct Option_Vec_f64 w_vector,
                            uintptr_t x_cols);

/**
 * Predict
 *
 * ## Parameters
 * * `x_vector`: A `Vec<f64>` of flattened features
 * * `w_vector`: A `Option<Vec<f64>>` of flattened sample weights
 * * `x_cols`: A usize corresponding the the number of features in preflattened X
 *
 * ## NOTE:
 *
 * It is probably a better idea to implement it as
 * `predict_numeric`, `predict_tree` etc.
 * - Or maybe an enum?
 */
double *predict(struct OpaquePointer *model_ptr, struct Vec_f64 x_vector, uintptr_t x_cols);

void free_perpetual_booster(struct OpaquePointer *model_ptr);

void free_predictions(double *ptr, uintptr_t length);
