# Get feature importance

Get feature importance

## Usage

``` r
perpetual_importance(model, method = "gain", normalize = TRUE)
```

## Arguments

- model:

  A `PerpetualBooster` object.

- method:

  String, method for importance (e.g. "gain").

- normalize:

  Boolean.

## Value

A named vector of importances.
