test_that("PerpetualBooster works", {
  X <- as.matrix(iris[, 1:4])
  y <- as.numeric(iris[, 5]) - 1 # 0, 1, 2
  
  # Binary classification (using only 2 classes for simplicity)
  y_bin <- ifelse(y == 2, 0, y)
  
  # Flatten matrix for Rust API (column-major)
  flat_data <- as.vector(X)
  rows <- nrow(X)
  cols <- ncol(X)
  
  model <- PerpetualBooster$new(objective = "LogLoss")
  model$fit(flat_data, rows, cols, y_bin)
  
  preds <- model$predict(flat_data, rows, cols)
  expect_true(length(preds) == rows)
  
  # Check saving and loading
  tmp <- tempfile(fileext = ".json")
  model$save_booster(tmp)
  
  model2 <- PerpetualBooster$load_booster(tmp)
  preds2 <- model2$predict(flat_data, rows, cols)
  expect_equal(preds, preds2)
})
