# Calibrate a PerpetualBooster model

Calibrate a PerpetualBooster model

## Usage

``` r
perpetual_calibrate(model, x, y, x_cal, y_cal, alpha, method = NULL)
```

## Arguments

- model:

  A `PerpetualBooster` object.

- x:

  Validation features.

- y:

  Validation targets.

- x_cal:

  Calibration features.

- y_cal:

  Calibration targets.

- alpha:

  Calibration parameter.

- method:

  String specifying the calibration method to use. If NULL, uses the
  method configured in the model.
