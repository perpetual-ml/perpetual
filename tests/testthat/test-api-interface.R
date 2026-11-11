# Tests for enhanced R API interface (LightGBM/XGBoost style)

test_that("perpetual_save and perpetual_load work as expected", {
  X <- as.matrix(iris[, 1:4])
  storage.mode(X) <- "double"
  y <- as.numeric(iris[, 5])
  
  model <- perpetual(X, y, objective = "LogLoss")
  preds1 <- predict(model, X, type = "prob")
  
  tmp <- tempfile(fileext = ".json")
  perpetual_save(model, tmp)
  
  expect_true(file.exists(tmp))
  
  model2 <- perpetual_load(tmp)
  preds2 <- predict(model2, X, type = "prob")
  
  expect_equal(preds1, preds2)
  expect_equal(attr(model, "classes"), attr(model2, "classes"))
  
  unlink(tmp)
})

test_that("predict with type = 'contribution' returns valid SHAP values", {
  X <- as.matrix(iris[1:10, 1:4])
  storage.mode(X) <- "double"
  y <- as.numeric(iris[1:10, 5])
  
  # Binary classification for simplicity
  y_bin <- ifelse(y == 1, 1, 0)
  
  model <- perpetual(X, y_bin, objective = "LogLoss", budget = 1.0)
  
  # Get contributions
  contribs <- predict(model, X, type = "contribution")
  
  expect_true(is.matrix(contribs))
  expect_equal(nrow(contribs), 10)
  expect_equal(ncol(contribs), 5) # 4 features + 1 bias
  expect_true(!any(is.na(contribs)))
})

test_that("predict.PerpetualBooster handles missing classes gracefully", {
  X <- as.matrix(iris[1:10, 1:4])
  storage.mode(X) <- "double"
  y <- as.numeric(iris[1:10, 5])
  
  model <- perpetual(X, y, objective = "SquaredLoss")
  # Manually strip classes to simulate a model loaded without meta (unlikely now but good to test)
  attr(model, "classes") <- NULL
  
  preds <- predict(model, X, type = "raw")
  expect_true(is.numeric(preds))
  expect_equal(length(preds), 10)
})

test_that("perpetual() training function handles additional parameters", {
  X <- as.matrix(iris[, 1:4])
  storage.mode(X) <- "double"
  y <- as.numeric(iris[, 5])
  
  # Test passing budget and other params - use full dataset to ensure trees are built
  model <- perpetual(X, y, objective = "LogLoss", budget = 1.0, max_bin = 128, log_iterations = 1)
  
  expect_s3_class(model, "PerpetualBooster")
  expect_true(perpetual_n_trees(model) > 0)
})
