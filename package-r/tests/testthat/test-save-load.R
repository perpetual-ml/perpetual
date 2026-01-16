test_that("Save and Load Booster works", {
  X <- as.matrix(iris[, 1:4])
  y <- as.numeric(iris[, 5]) - 1
  y_bin <- ifelse(y == 2, 0, y)
  
  flat_data <- as.vector(X)
  rows <- nrow(X)
  cols <- ncol(X)
  
  model <- PerpetualBooster$new(objective = "LogLoss")
  model$fit(flat_data, rows, cols, y_bin)
  preds1 <- model$predict(flat_data, rows, cols)
  
  tmp <- tempfile(fileext = ".json")
  model$save_booster(tmp)
  
  model2 <- PerpetualBooster$load_booster(tmp)
  preds2 <- model2$predict(flat_data, rows, cols)
  
  expect_equal(preds1, preds2)
  expect_true(file.exists(tmp))
})
