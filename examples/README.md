# Perpetual R Package Examples

This directory contains runnable examples demonstrating the Perpetual gradient boosting library in R.
All examples use R's built-in datasets, so you can copy and run them immediately without any setup.

## Prerequisites

Install the perpetual package:

```r
# Install from source (from package-r directory)
install.packages(".", repos = NULL, type = "source")

# Or install rextendr and build
# install.packages("rextendr")
# rextendr::document()
```

## Examples

| File | Description | Dataset | Task |
| ---- | ----------- | ------- | ---- |
| `01_binary_classification.R` | Binary classification | `mtcars` | Predict transmission type |
| `02_multiclass_classification.R` | Multiclass classification | `iris` | Predict flower species |
| `03_regression.R` | Regression | `mtcars` | Predict fuel efficiency (mpg) |
| `04_feature_importance.R` | Feature importance analysis | `mtcars` | Analyze feature contributions |
| `05_shap_contributions.R` | SHAP-style contributions | `iris` | Explain individual predictions |
| `06_save_load_model.R` | Model persistence | `mtcars` | Save and load models |
| `07_prediction_intervals.R` | Regression with airquality | `airquality` | Predict ozone levels |
| `08_budget_comparison.R` | Effect of training budget | `airquality` | Compare model complexity |

## Running Examples

Simply source any example file:

```r
source("examples/01_binary_classification.R")
```

Or run interactively in RStudio by opening the file and pressing Ctrl+Shift+Enter.

## Key Functions

- `perpetual()` - Train a model
- `predict()` - Make predictions (supports types: `"class"`, `"prob"`, `"raw"`, `"contribution"`)
- `perpetual_save()` / `perpetual_load()` - Model persistence
- `perpetual_importance()` - Feature importance

## Objectives

- `"LogLoss"` - Binary/multiclass classification
- `"SquaredLoss"` - Regression
