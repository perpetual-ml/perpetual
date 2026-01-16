test_that("Multiclass prediction works", {
  X <- as.matrix(iris[, 1:4])
  y <- as.numeric(iris[, 5]) - 1 # 0, 1, 2 (3 classes)
  
  flat_data <- as.vector(X)
  rows <- nrow(X)
  cols <- ncol(X)
  
  model <- PerpetualBooster$new(objective = "LogLoss")
  model$fit(flat_data, rows, cols, y)
  
  preds <- model$predict(flat_data, rows, cols)
  expect_true(length(preds) == rows)
  expect_true(!any(is.na(preds)))
  
  # Check predict_proba returns probabilities
  # PerpetualBooster's predict_proba returns one probability per sample 
  # (sigmoid transformation of predictions)
  preds_proba <- model$predict_proba(flat_data, rows, cols)
  expect_true(length(preds_proba) == rows)
  # Probabilities should be between 0 and 1
  expect_true(all(preds_proba >= 0 & preds_proba <= 1))
})
