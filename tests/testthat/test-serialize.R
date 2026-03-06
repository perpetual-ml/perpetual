# Serialize tests replicating Python test_serialize.py
# R implementation using jsonlite for JSON serialization

test_that("Scalar values serialize/deserialize correctly", {
  values <- list(
    1L,
    1.0,
    1.00101,
    "a string",
    TRUE,
    FALSE,
    NULL
  )
  
  for (value in values) {
    json_str <- jsonlite::toJSON(value, auto_unbox = TRUE, null = "null")
    expect_true(is.character(json_str))
    
    restored <- jsonlite::fromJSON(json_str)
    if (is.null(value)) {
      expect_null(restored)
    } else if (is.logical(value)) {
      expect_equal(as.logical(restored), value)
    } else if (is.character(value)) {
      expect_equal(as.character(restored), value)
    } else {
      # Use tolerance for numeric comparison as JSON roundtrip might change precision
      expect_equal(as.numeric(restored), as.numeric(value), tolerance = 1e-4)
    }
  }
})

test_that("Object values serialize/deserialize correctly", {
  values <- list(
    c(1, 2, 3),
    c(1.0, 4.0),
    c("a", "b", "c"),
    list(a = 1.0, b = 2.0),
    list(a = "test", b = "what")
  )
  
  for (value in values) {
    json_str <- jsonlite::toJSON(value, auto_unbox = TRUE)
    expect_true(is.character(json_str))
    
    restored <- jsonlite::fromJSON(json_str)
    
    if (is.list(value) && !is.null(names(value))) {
      # Named list (dict-like)
      expect_equal(as.list(restored), value, tolerance = 1e-10)
    } else {
      # Vector or unnamed list
      expect_equal(as.vector(restored), as.vector(value), tolerance = 1e-10)
    }
  }
})

test_that("Numeric arrays serialize/deserialize correctly", {
  values <- list(
    c(1.0, 2.23),
    matrix(1:6, nrow = 2, ncol = 3),
    matrix(as.integer(1:6), nrow = 2, ncol = 3)
  )
  
  for (value in values) {
    json_str <- jsonlite::toJSON(value)
    expect_true(is.character(json_str))
    
    restored <- jsonlite::fromJSON(json_str)
    
    if (is.matrix(value)) {
      restored <- matrix(restored, nrow = nrow(value), ncol = ncol(value))
    }
    
    expect_equal(as.vector(restored), as.vector(value), tolerance = 1e-10)
  }
})

test_that("Model JSON contains expected structure", {
  data <- as_matrix_data(load_titanic_test_data())
  
  model <- perpetual(data$X, data$y, objective = "LogLoss")
  
  json_str <- perpetual_to_json(model)
  parsed <- jsonlite::fromJSON(json_str, simplifyVector = FALSE)
  
  # Check expected fields exist
  expect_true("trees" %in% names(parsed))
  expect_true("base_score" %in% names(parsed))
  expect_true("cfg" %in% names(parsed))
})
