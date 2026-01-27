# Predict using a PerpetualBooster model

Predict using a PerpetualBooster model

## Usage

``` r
# S3 method for class 'PerpetualBooster'
predict(
  object,
  newdata,
  type = c("class", "prob", "raw", "contribution", "interval"),
  method = "average",
  ...
)
```

## Arguments

- object:

  A `PerpetualBooster` object.

- newdata:

  A matrix or data.frame of new data to predict on.

- type:

  Type of prediction: "class", "prob", "raw", "contribution", or
  "interval".

- method:

  Method for prediction contributions. Default "average".

- ...:

  Additional arguments.

## Value

A vector or matrix of predictions.
