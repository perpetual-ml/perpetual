# Contribution tests using XGBoost/LightGBM style API
# Replicating Python test_booster.py contribution tests

test_that("predict(type='contribution') returns correct shape", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  contribs <- predict(model, data$X, type = "contribution", method = "average")
  
  # Should return (n_samples x (n_features + 1)) matrix - last column is bias
  expect_true(is.matrix(contribs))
  expect_equal(nrow(contribs), nrow(data$X))
  expect_equal(ncol(contribs), ncol(data$X) + 1)
})

test_that("predict(type='contribution') average method returns valid values", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  contribs <- predict(model, data$X, type = "contribution", method = "average")
  
  expect_true(is.matrix(contribs))
  expect_true(!any(is.na(contribs)))
  expect_true(all(is.finite(contribs)))
})

test_that("predict(type='contribution') weight method works", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  contribs <- predict(model, data$X, type = "contribution", method = "weight")
  
  expect_true(is.matrix(contribs))
  expect_true(!any(is.na(contribs)))
})

test_that("predict(type='contribution') branchdifference method works", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  contribs <- predict(model, data$X, type = "contribution", method = "branchdifference")
  
  expect_true(is.matrix(contribs))
  expect_true(!any(is.na(contribs)))
})

test_that("predict(type='contribution') shapley method works", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  contribs <- predict(model, data$X, type = "contribution", method = "shapley")
  
  expect_true(is.matrix(contribs))
  expect_true(!any(is.na(contribs)))
})

test_that("predict(type='contribution') probabilitychange method works for LogLoss", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  contribs <- predict(model, data$X, type = "contribution", method = "probabilitychange")
  
  expect_true(is.matrix(contribs))
  expect_true(!any(is.na(contribs)))
})
