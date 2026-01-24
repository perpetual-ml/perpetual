# Regression tests with cal_housing data using XGBoost/LightGBM style API

test_that("Regression with cal_housing data works", {
  skip_if_not(file.exists(file.path(get_resources_dir(), "cal_housing_train.csv")),
              "Cal housing data not available")
  
  data <- load_cal_housing_data()
  X_train <- matrix(data$train$flat_data, nrow = data$train$rows, ncol = data$train$cols, byrow = FALSE)
  X_test <- matrix(data$test$flat_data, nrow = data$test$rows, ncol = data$test$cols, byrow = FALSE)
  storage.mode(X_train) <- "double"
  storage.mode(X_test) <- "double"
  
  model <- perpetual(X_train, data$train$y, objective = "SquaredLoss")
  
  preds <- predict(model, X_test, type = "raw")
  
  expect_true(length(preds) == data$test$rows)
  expect_true(!any(is.na(preds)))
  expect_true(is.numeric(preds))
})

test_that("Regression predictions are reasonable", {
  skip_if_not(file.exists(file.path(get_resources_dir(), "cal_housing_train.csv")),
              "Cal housing data not available")
  
  data <- load_cal_housing_data()
  X_train <- matrix(data$train$flat_data, nrow = data$train$rows, ncol = data$train$cols, byrow = FALSE)
  X_test <- matrix(data$test$flat_data, nrow = data$test$rows, ncol = data$test$cols, byrow = FALSE)
  storage.mode(X_train) <- "double"
  storage.mode(X_test) <- "double"
  
  model <- perpetual(X_train, data$train$y, objective = "SquaredLoss")
  
  preds <- predict(model, X_test, type = "raw")
  
  # Predictions should be in a reasonable range (house values are between 0.5 and 5+)
  expect_true(all(preds > 0))
  expect_true(all(preds < 10))  # Median house values in 100k units
  
  # Should have some variance (not all the same prediction)
  expect_true(sd(preds) > 0)
})

test_that("Regression save/load preserves predictions", {
  skip_if_not(file.exists(file.path(get_resources_dir(), "cal_housing_train.csv")),
              "Cal housing data not available")
  
  data <- load_cal_housing_data()
  X_train <- matrix(data$train$flat_data, nrow = data$train$rows, ncol = data$train$cols, byrow = FALSE)
  X_test <- matrix(data$test$flat_data, nrow = data$test$rows, ncol = data$test$cols, byrow = FALSE)
  storage.mode(X_train) <- "double"
  storage.mode(X_test) <- "double"
  
  model <- perpetual(X_train, data$train$y, objective = "SquaredLoss", iteration_limit = 20L)
  preds1 <- predict(model, X_test, type = "raw")
  
  tmp <- tempfile(fileext = ".json")
  perpetual_save(model, tmp)
  
  model2 <- perpetual_load(tmp)
  preds2 <- predict(model2, X_test, type = "raw")
  
  expect_equal(preds1, preds2)
  
  unlink(tmp)
})
