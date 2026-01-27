# Helper functions to load test data from resources directory
# These match the data used in Python tests

library(perpetual)

get_resources_dir <- function() {
  # During tests, resources are copied to tests/testthat/resources
  # testthat tests run with getwd() as tests/testthat (usually) or tests/
  
  candidates <- c(
    "resources",
    "tests/testthat/resources",
    "../resources",
    "../../resources",
    "./testthat/resources"
  )
  
  for (path in candidates) {
    if (dir.exists(path)) {
      # Normalize path
      return(normalizePath(path))
    }
  }
  
  # Failback or check if we are in package root (manually run)
  
  # Debug info for CI
  # msg <- paste0(
  #   "Could not find resources directory.\n",
  #   "Working directory: ", getwd(), "\n",
  #   "Candidate paths checked: ", paste(candidates, collapse = ", "), "\n",
  #   "Files in current dir:\n", paste(list.files(), collapse = "\n"), "\n",
  #   "Files in parent dir:\n", paste(list.files(".."), collapse = "\n")
  # )
  
  # stop(msg)
  return(NULL)
}

skip_if_no_resources <- function() {
  if (is.null(get_resources_dir())) {
    if (Sys.getenv("PERPETUAL_REQ_RESOURCES") == "true") {
      stop("Strict tests enabled (CI): Resources not found but required!")
    }
    testthat::skip("Test resources not found (skipped on R-universe/CRAN without data)")
  }
}

# Helper to convert flat data list to matrix for new API
as_matrix_data <- function(data) {
  X <- matrix(data$flat_data, nrow = data$rows, ncol = data$cols, byrow = FALSE)
  storage.mode(X) <- "double"
  list(X = X, y = data$y)
}

# Load Titanic test data (matches Python test_booster.py test_categorical)
load_titanic_test_data <- function() {
  skip_if_no_resources()
  resources_dir <- get_resources_dir()
  
  # Read the flat CSV (column-major values)
  flat_file <- file.path(resources_dir, "titanic_test_flat.csv")
  y_file <- file.path(resources_dir, "titanic_test_y.csv")
  
  lines_flat <- readLines(flat_file)
  lines_flat <- lines_flat[grep("\\S", lines_flat)]
  lines_flat[lines_flat == "\"\"" | lines_flat == ""] <- NA
  flat_data <- as.numeric(lines_flat)
  
  lines_y <- readLines(y_file)
  lines_y <- lines_y[grep("\\S", lines_y)]
  lines_y[lines_y == "\"\"" | lines_y == ""] <- NA
  y <- as.numeric(lines_y)
  
  # titanic_test has 178 rows, need to determine cols
  rows <- length(y)
  cols <- length(flat_data) / rows
  
  list(flat_data = flat_data, rows = rows, cols = as.integer(cols), y = y)
}

# Load Titanic training data
load_titanic_train_data <- function() {
  skip_if_no_resources()
  resources_dir <- get_resources_dir()
  
  flat_file <- file.path(resources_dir, "titanic_train_flat.csv")
  y_file <- file.path(resources_dir, "titanic_train_y.csv")
  
  lines_flat <- readLines(flat_file)
  lines_flat <- lines_flat[grep("\\S", lines_flat)]
  lines_flat[lines_flat == "\"\"" | lines_flat == ""] <- NA
  flat_data <- as.numeric(lines_flat)
  
  lines_y <- readLines(y_file)
  lines_y <- lines_y[grep("\\S", lines_y)]
  lines_y[lines_y == "\"\"" | lines_y == ""] <- NA
  y <- as.numeric(lines_y)
  
  rows <- length(y)
  cols <- length(flat_data) / rows
  
  list(flat_data = flat_data, rows = rows, cols = as.integer(cols), y = y)
}

# Load Cover Types data for multi-class classification (matches Python test_multi_output.py)
load_cover_types_data <- function(n_samples = 1000, seed = 0) {
  skip_if_no_resources()
  resources_dir <- get_resources_dir()
  
  train_file <- file.path(resources_dir, "cover_types_train.csv")
  test_file <- file.path(resources_dir, "cover_types_test.csv")
  
  # Read CSV with header
  train_df <- read.csv(train_file, header = TRUE)
  test_df <- read.csv(test_file, header = TRUE)
  
  # Sample training data like Python test
  set.seed(seed)
  if (nrow(train_df) > n_samples) {
    train_df <- train_df[sample(nrow(train_df), n_samples), ]
  }
  
  # Extract y (Cover_Type column)
  y_train <- train_df$Cover_Type
  y_test <- test_df$Cover_Type
  
  # Remove target column for X
  X_train <- train_df[, !names(train_df) %in% c("Cover_Type")]
  X_test <- test_df[, !names(test_df) %in% c("Cover_Type")]
  
  # Convert to numeric matrix and flatten (column-major for R)
  X_train_mat <- as.matrix(X_train)
  storage.mode(X_train_mat) <- "double"
  X_test_mat <- as.matrix(X_test)
  storage.mode(X_test_mat) <- "double"
  
  list(
    train = list(
      flat_data = as.vector(X_train_mat),
      rows = nrow(X_train_mat),
      cols = ncol(X_train_mat),
      y = as.numeric(y_train)
    ),
    test = list(
      flat_data = as.vector(X_test_mat),
      rows = nrow(X_test_mat),
      cols = ncol(X_test_mat),
      y = as.numeric(y_test)
    )
  )
}

# Load California Housing data for regression (matches Python test_calibration)
load_cal_housing_data <- function() {
  skip_if_no_resources()
  resources_dir <- get_resources_dir()
  
  train_file <- file.path(resources_dir, "cal_housing_train.csv")
  test_file <- file.path(resources_dir, "cal_housing_test.csv")
  
  # Read CSV with header
  train_df <- read.csv(train_file, header = TRUE)
  test_df <- read.csv(test_file, header = TRUE)
  
  # Target column is MedHouseVal
  y_train <- train_df$MedHouseVal
  y_test <- test_df$MedHouseVal
  
  # Remove target column for X
  X_train <- train_df[, !names(train_df) %in% c("MedHouseVal")]
  X_test <- test_df[, !names(test_df) %in% c("MedHouseVal")]
  
  # Convert to numeric matrix and flatten
  X_train_mat <- as.matrix(X_train)
  X_test_mat <- as.matrix(X_test)
  
  list(
    train = list(
      flat_data = as.vector(X_train_mat),
      rows = nrow(X_train_mat),
      cols = ncol(X_train_mat),
      y = as.numeric(y_train)
    ),
    test = list(
      flat_data = as.vector(X_test_mat),
      rows = nrow(X_test_mat),
      cols = ncol(X_test_mat),
      y = as.numeric(y_test)
    )
  )
}
