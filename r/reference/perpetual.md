# Train a PerpetualBooster model

Perpetual is a self-generalizing gradient boosting machine that doesn't
need hyperparameter optimization. It automatically finds the best
configuration based on the provided budget.

## Usage

``` r
perpetual(
  x,
  y,
  objective = "LogLoss",
  budget = NULL,
  iteration_limit = NULL,
  stopping_rounds = NULL,
  max_bin = NULL,
  num_threads = NULL,
  missing = NULL,
  allow_missing_splits = NULL,
  create_missing_branch = NULL,
  missing_node_treatment = NULL,
  log_iterations = NULL,
  quantile = NULL,
  reset = NULL,
  timeout = NULL,
  memory_limit = NULL,
  seed = NULL,
  calibration_method = NULL,
  save_node_stats = NULL,
  ...
)
```

## Arguments

- x:

  A matrix or data.frame of features.

- y:

  A vector of targets (numeric for regression, factor/integer for
  classification).

- objective:

  A string specifying the objective function. Default is "LogLoss".

- budget:

  A numeric value ensuring the training time does not exceed this budget
  (in normalized units).

- iteration_limit:

  An integer limit on the number of iterations.

- stopping_rounds:

  An integer for early stopping.

- max_bin:

  Integer, max number of bins for histograms.

- num_threads:

  Integer, number of threads to use.

- missing:

  Value to consider as missing data. Default is NaN.

- allow_missing_splits:

  Boolean.

- create_missing_branch:

  Boolean. Whether to create a separate branch for missing values
  (ternary trees).

- missing_node_treatment:

  String. How to handle weights for missing nodes if
  create_missing_branch is True. Options: "None", "AssignToParent",
  "AverageLeafWeight", "AverageNodeWeight".

- log_iterations:

  Integer.

- quantile:

  Numeric.

- reset:

  Boolean.

- timeout:

  Numeric.

- memory_limit:

  Numeric.

- seed:

  Integer seed for reproducibility.

- calibration_method:

  String specifying the calibration method for prediction intervals.
  Options: "WeightVariance", "MinMax", "GRP", "Conformal".

- save_node_stats:

  Boolean. Whether to save node statistics (required for some
  calibration methods and importance types).

- ...:

  Additional arguments.

## Value

A `PerpetualBooster` object.
