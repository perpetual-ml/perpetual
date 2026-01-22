# Save and load tests using XGBoost/LightGBM style API
# Replicating Python test_save_load.py

test_that("Save and Load Booster works with binary classification (Iris)", {
  X <- as.matrix(iris[, 1:4])
  storage.mode(X) <- "double"
  y <- as.numeric(iris[, 5])
  y_bin <- ifelse(y == 3, 0, ifelse(y == 2, 1, 0))  # Binary: 0 or 1
  
  model <- perpetual(X, y_bin, objective = "LogLoss")
  preds1 <- predict(model, X, type = "raw")
  
  tmp <- tempfile(fileext = ".json")
  perpetual_save(model, tmp)
  
  model2 <- perpetual_load(tmp)
  preds2 <- predict(model2, X, type = "raw")
  
  expect_equal(preds1, preds2)
  expect_true(file.exists(tmp))
  
  unlink(tmp)
})

test_that("Save and Load Booster works with multiclass (Iris)", {
  X <- as.matrix(iris[, 1:4])
  storage.mode(X) <- "double"
  y <- as.numeric(iris[, 5])  # 1, 2, 3
  
  model <- perpetual(X, y, objective = "LogLoss")
  preds1 <- predict(model, X, type = "class")
  
  tmp <- tempfile(fileext = ".json")
  perpetual_save(model, tmp)
  
  model2 <- perpetual_load(tmp)
  preds2 <- predict(model2, X, type = "class")
  
  expect_equal(preds1, preds2)
  
  unlink(tmp)
})

test_that("Save and load with SquaredLoss preserves predictions (Titanic)", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "SquaredLoss", iteration_limit = 10L)
  preds1 <- predict(model, data$X, type = "raw")
  
  # Save
  tmp <- tempfile(fileext = ".json")
  perpetual_save(model, tmp)
  expect_true(file.exists(tmp))
  
  # Load
  model2 <- perpetual_load(tmp)
  preds2 <- predict(model2, data$X, type = "raw")
  
  expect_equal(preds1, preds2)
  
  # Cleanup
  unlink(tmp)
})

test_that("Save and load with LogLoss preserves predictions (Titanic)", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss", iteration_limit = 10L)
  preds1 <- predict(model, data$X, type = "raw")
  proba1 <- predict(model, data$X, type = "prob")
  
  # Save
  tmp <- tempfile(fileext = ".json")
  perpetual_save(model, tmp)
  
  # Load
  model2 <- perpetual_load(tmp)
  preds2 <- predict(model2, data$X, type = "raw")
  proba2 <- predict(model2, data$X, type = "prob")
  
  expect_equal(preds1, preds2)
  expect_equal(proba1, proba2)
  
  unlink(tmp)
})

test_that("Saved model JSON is valid", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  tmp <- tempfile(fileext = ".json")
  perpetual_save(model, tmp)
  
  # Read and parse JSON
  json_content <- readLines(tmp, warn = FALSE)
  json_str <- paste(json_content, collapse = "\n")
  parsed <- jsonlite::fromJSON(json_str, simplifyVector = FALSE)
  
  expect_true("trees" %in% names(parsed))
  expect_true("base_score" %in% names(parsed))
  
  unlink(tmp)
})

test_that("number_of_trees and base_score preserved after load", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  n_trees1 <- model$number_of_trees()
  base1 <- model$base_score()
  
  tmp <- tempfile(fileext = ".json")
  perpetual_save(model, tmp)
  
  model2 <- perpetual_load(tmp)
  
  expect_equal(n_trees1, model2$number_of_trees())
  expect_equal(base1, model2$base_score())
  
  unlink(tmp)
})
