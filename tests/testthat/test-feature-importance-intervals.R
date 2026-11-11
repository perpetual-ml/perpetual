# Test feature importance and prediction intervals
# These are new features added to the Perpetual R package

test_that("perpetual_importance returns feature importance scores", {
  X <- as.matrix(iris[, 1:4])
  y <- as.numeric(iris[, 5])
  storage.mode(X) <- "double"
  
  model <- perpetual(X, y, objective = "LogLoss", budget = 0.5, seed = 123)
  
  importance <- perpetual_importance(model, method = "gain")
  
  expect_true(length(importance) == 4)
  expect_true(all(importance >= 0))
  expect_true(is.numeric(importance))
})

test_that("perpetual supports seed parameter for reproducibility", {
  X <- as.matrix(iris[, 1:4])
  y <- as.numeric(iris[, 5])
  storage.mode(X) <- "double"
  
  model1 <- perpetual(X, y, objective = "LogLoss", budget = 0.5, seed = 42, num_threads = 1L)
  model2 <- perpetual(X, y, objective = "LogLoss", budget = 0.5, seed = 42, num_threads = 1L)
  
  preds1 <- predict(model1, X, type = "raw")
  preds2 <- predict(model2, X, type = "raw")
  
  expect_equal(preds1, preds2)
})

test_that("perpetual_calibrate and predict intervals work", {
  y_reg <- iris[, 1]
  X_reg <- as.matrix(iris[, 2:4])
  storage.mode(X_reg) <- "double"
  
  model_reg <- perpetual(X_reg, y_reg, objective = "SquaredLoss", budget = 0.5, save_node_stats = TRUE)
  
  # Split for calibration
  set.seed(123)
  idx <- sample(1:nrow(iris), 100)
  X_train <- X_reg[idx, , drop = FALSE]
  y_train <- y_reg[idx]
  X_cal <- X_reg[-idx, , drop = FALSE]
  y_cal <- y_reg[-idx]
  
  perpetual_calibrate(model_reg, X_train, y_train, X_cal, y_cal, alpha = c(0.1))
  
  X_test <- X_reg[1:5, , drop = FALSE]
  intervals <- predict(model_reg, X_test, type = "interval")
  
  expect_true(!is.null(intervals))
  expect_true(length(intervals) > 0)
  expect_true("0.1" %in% names(intervals))
})
