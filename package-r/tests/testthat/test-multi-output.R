# Multiclass classification tests using XGBoost/LightGBM style API

test_that("Multiclass prediction with perpetual() and predict()", {
  X <- as.matrix(iris[, 1:4])
  storage.mode(X) <- "double"
  y <- as.numeric(iris[, 5])  # 1, 2, 3 (3 classes)
  
  model <- perpetual(X, y, objective = "LogLoss")
  
  # Class predictions should return labels for each sample
  preds_class <- predict(model, X, type = "class")
  expect_true(length(preds_class) == nrow(X))
  expect_true(all(preds_class %in% c(1, 2, 3)))
  
  # Probability predictions should return matrix (n_samples x n_classes)
  preds_prob <- predict(model, X, type = "prob")
  expect_true(is.matrix(preds_prob))
  expect_equal(nrow(preds_prob), nrow(X))
  expect_equal(ncol(preds_prob), 3)  # 3 classes
  
  # Probabilities should be between 0 and 1
  expect_true(all(preds_prob >= 0 & preds_prob <= 1))
  
  # Each row should sum to 1
  row_sums <- rowSums(preds_prob)
  expect_equal(row_sums, rep(1, nrow(X)), tolerance = 1e-6)
  
  # Raw predictions
  preds_raw <- predict(model, X, type = "raw")
  expect_true(length(preds_raw) == nrow(X) * 3)  # n_samples * n_classes
})
