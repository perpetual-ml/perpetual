# Basic booster tests replicating Python test_booster.py
# Uses XGBoost/LightGBM style API: perpetual(), predict()

test_that("Basic fit and predict works with titanic data", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  preds <- predict(model, data$X, type = "raw")
  expect_true(length(preds) == nrow(data$X))
  expect_true(!any(is.na(preds)))
})

test_that("predict with type='prob' returns valid probabilities", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  proba <- predict(model, data$X, type = "prob")
  expect_true(length(proba) == nrow(data$X))
  # Probabilities should be between 0 and 1
  expect_true(all(proba >= 0 & proba <= 1))
})

test_that("predict with type='class' returns labels", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  labels <- predict(model, data$X, type = "class")
  expect_true(length(labels) == nrow(data$X))
  # Labels should be 0 or 1 for binary classification
  expect_true(all(labels %in% c(0, 1)))
})

test_that("Multiple model trains produce consistent results", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model1 <- perpetual(data$X, data$y, objective = "LogLoss", num_threads = 1L)
  preds1 <- predict(model1, data$X, type = "raw")
  
  # Fit again with same params
  model2 <- perpetual(data$X, data$y, objective = "LogLoss", num_threads = 1L)
  preds2 <- predict(model2, data$X, type = "raw")
  
  expect_equal(preds1, preds2, tolerance = 1e-6)
})

test_that("number_of_trees returns positive integer", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  n_trees <- model$number_of_trees()
  expect_true(is.numeric(n_trees))
  expect_true(n_trees > 0)
})

test_that("base_score returns numeric value", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  base <- model$base_score()
  expect_true(is.numeric(base))
})

test_that("json_dump returns valid JSON", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  json_str <- model$json_dump()
  expect_true(nchar(json_str) > 0)
  
  # Should be valid JSON
  parsed <- jsonlite::fromJSON(json_str, simplifyVector = FALSE)
  expect_true(!is.null(parsed))
  expect_true("trees" %in% names(parsed))
})

test_that("SquaredLoss objective works for regression", {
  # Use titanic data with numeric target for regression
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "SquaredLoss")
  
  preds <- predict(model, data$X, type = "raw")
  expect_true(length(preds) == nrow(data$X))
  expect_true(!any(is.na(preds)))
})

test_that("Different booster parameters work", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(
    data$X, data$y,
    objective = "LogLoss",
    budget = 0.3,
    max_bin = 128L,
    num_threads = 1L,
    iteration_limit = 10L
  )
  
  preds <- predict(model, data$X, type = "raw")
  expect_true(length(preds) == nrow(data$X))
})

test_that("print() displays model summary", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  # Should not error and print output
  output <- capture.output(print(model))
  expect_true(any(grepl("Perpetual", output)))
})
