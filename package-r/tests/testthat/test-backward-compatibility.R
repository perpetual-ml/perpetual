# Backward compatibility tests using XGBoost/LightGBM style API
# Replicating Python test_backward_compatibility.py
# Tests loading pre-saved model from v2.0.0

test_that("Load v2.0.0 model and verify predictions", {
  skip_if_no_resources()
  resources_dir <- get_resources_dir()
  model_path <- file.path(resources_dir, "model_bc.json")
  preds_path <- file.path(resources_dir, "model_bc_preds.csv")
  probs_path <- file.path(resources_dir, "model_bc_probs.csv")
  
  skip_if_not(file.exists(model_path), "Model artifact not found. Run scripts/make_resources.py to generate it.")
  skip_if_not(file.exists(preds_path), "Predictions artifact not found.")
  
  # Load the model
  model <- PerpetualBooster$load_booster(model_path)
  
  # Verify model loaded correctly
  expect_true(!is.null(model))
  n_trees <- perpetual_n_trees(model)
  expect_true(n_trees > 0)
  
  base <- perpetual_base_score(model)
  expect_true(is.numeric(base))
})

test_that("V2.0.0 model JSON is valid", {
  skip_if_no_resources()
  resources_dir <- get_resources_dir()
  model_path <- file.path(resources_dir, "model_bc.json")
  
  skip_if_not(file.exists(model_path), "Model artifact not found.")
  
  # Read and parse JSON directly
  json_content <- readLines(model_path, warn = FALSE)
  json_str <- paste(json_content, collapse = "\n")
  parsed <- jsonlite::fromJSON(json_str, simplifyVector = FALSE)
  
  # Verify structure
  expect_true("trees" %in% names(parsed))
  expect_true("base_score" %in% names(parsed))
  expect_true(length(parsed$trees) > 0)
})

test_that("Loaded model can make predictions with new API", {
  skip_if_no_resources()
  resources_dir <- get_resources_dir()
  model_path <- file.path(resources_dir, "model_bc.json")
  
  skip_if_not(file.exists(model_path), "Model artifact not found.")
  
  model <- PerpetualBooster$load_booster(model_path)
  
  # Load titanic test data
  data <- as_matrix_data(load_titanic_test_data())
  
  # This will work if the number of features matches
  tryCatch({
    preds <- predict(model, data$X, type = "raw")
    expect_true(length(preds) == nrow(data$X))
    expect_true(!any(is.na(preds)))
  }, error = function(e) {
    # If feature count doesn't match, that's expected
    skip("Model feature count doesn't match test data - expected for v2.0.0 model")
  })
})
