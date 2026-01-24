#' Train a PerpetualBooster model
#'
#' Perpetual is a self-generalizing gradient boosting machine that doesn't need 
#' hyperparameter optimization. It automatically finds the best configuration 
#' based on the provided budget.
#'
#' @param x A matrix or data.frame of features.
#' @param y A vector of targets (numeric for regression, factor/integer for classification).
#' @param objective A string specifying the objective function. Default is "LogLoss".
#' @param budget A numeric value ensuring the training time does not exceed this budget (in normalized units).
#' @param iteration_limit An integer limit on the number of iterations.
#' @param stopping_rounds An integer for early stopping.
#' @param max_bin Integer, max number of bins for histograms.
#' @param num_threads Integer, number of threads to use.
#' @param missing Check documentation.
#' @param allow_missing_splits Boolean.
#' @param create_missing_branch Boolean.
#' @param missing_node_treatment String.
#' @param log_iterations Integer.
#' @param quantile Numeric.
#' @param reset Boolean.
#' @param timeout Numeric.
#' @param memory_limit Numeric.
#' @param seed Integer seed for reproducibility.
#' @param ... Additional arguments.
#'
#' @return A \code{PerpetualBooster} object.
#' @export
perpetual <- function(x, y, objective = "LogLoss", budget = NULL, 
                      iteration_limit = NULL, stopping_rounds = NULL, 
                      max_bin = NULL, num_threads = NULL, missing = NULL,
                      allow_missing_splits = NULL, create_missing_branch = NULL,
                      missing_node_treatment = NULL, log_iterations = NULL,
                      quantile = NULL, reset = NULL, timeout = NULL,
                      memory_limit = NULL, seed = NULL, ...) {
  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  
  # Flatten and types for Rust
  storage.mode(x) <- "double"
  flat_data <- as.vector(x)
  rows <- nrow(x)
  cols <- ncol(x)
  y <- as.numeric(y)
  
  # Detect classes for classification objectives
  classes <- sort(unique(y[!is.na(y)]))
  
  ptr <- .Call("PerpetualBooster_new",
        objective,
        budget,
        max_bin,
        num_threads,
        missing,
        allow_missing_splits,
        create_missing_branch,
        missing_node_treatment,
        log_iterations,
        quantile,
        reset,
        timeout,
        iteration_limit,
        memory_limit,
        stopping_rounds,
        seed,
        PACKAGE = "perpetual"
  )
  
  # Create a lightweight R6-like object (just a list with class)
  model <- structure(list(.ptr = ptr), class = "PerpetualBooster")
  
  # Fit
  .Call("PerpetualBooster_fit", ptr, flat_data, as.integer(rows), as.integer(cols), y, PACKAGE = "perpetual")
  
  # Store classes and objective
  attr(model, "classes") <- classes
  attr(model, "objective") <- objective
  
  return(model)
}

#' Predict using a PerpetualBooster model
#'
#' @param object A \code{PerpetualBooster} object.
#' @param newdata A matrix or data.frame of new data to predict on.
#' @param type Type of prediction: "class", "prob", "raw", "contribution", or "interval".
#' @param method Method for prediction contributions. Default "average".
#' @param ... Additional arguments.
#'
#' @return A vector or matrix of predictions.
#' @export
predict.PerpetualBooster <- function(object, newdata, type = c("class", "prob", "raw", "contribution", "interval"), method = "average", ...) {
  type <- match.arg(type)
  
  if (is.data.frame(newdata)) {
    newdata <- as.matrix(newdata)
  }
  storage.mode(newdata) <- "double"
  flat_data <- as.vector(newdata)
  rows <- nrow(newdata)
  cols <- ncol(newdata)
  
  raw_preds <- .Call("PerpetualBooster_predict", object$.ptr, flat_data, as.integer(rows), as.integer(cols), PACKAGE = "perpetual")
  
  classes <- attr(object, "classes")
  objective <- attr(object, "objective")
  
  if (type == "raw") {
    return(raw_preds)
  }
  
  if (type == "contribution") {
    contribs <- .Call("PerpetualBooster_predict_contributions", object$.ptr, flat_data, as.integer(rows), as.integer(cols), method, PACKAGE = "perpetual")
    contrib_mat <- matrix(contribs, nrow = rows, ncol = cols + 1, byrow = FALSE)
    return(contrib_mat)
  }
  
  if (type == "interval") {
    intervals <- .Call("PerpetualBooster_predict_intervals", object$.ptr, flat_data, as.integer(rows), as.integer(cols), PACKAGE = "perpetual")
    return(intervals)
  }
  
  if (is.null(classes) || length(classes) == 0) {
    if (type == "prob") {
      probs <- 1 / (1 + exp(-raw_preds))
      return(probs)
    } else {
      return(raw_preds)
    }
  }
  
  if (length(classes) <= 2) {
    if (length(classes) == 2 && !is.null(objective) && objective == "LogLoss") {
      probs <- 1 / (1 + exp(-raw_preds))
      if (type == "prob") {
        return(probs)
      } else {
        return(ifelse(probs >= 0.5, classes[2], classes[1]))
      }
    } else {
      return(raw_preds)
    }
  } else {
    n_classes <- length(classes)
    if (type == "prob") {
      probs_flat <- .Call("PerpetualBooster_predict_proba", object$.ptr, flat_data, as.integer(rows), as.integer(cols), PACKAGE = "perpetual")
      probs <- matrix(probs_flat, nrow = rows, ncol = n_classes, byrow = FALSE)
      row_sums <- rowSums(probs)
      probs <- probs / row_sums
      return(probs)
    } else {
      preds_mat <- matrix(raw_preds, nrow = rows, ncol = n_classes, byrow = FALSE)
      max_idx <- max.col(preds_mat, ties.method = "first")
      return(classes[max_idx])
    }
  }
}

#' Print PerpetualBooster
#'
#' @param x A \code{PerpetualBooster} object.
#' @param ... Additional arguments.
#'
#' @export
print.PerpetualBooster <- function(x, ...) {
  n_trees <- .Call("PerpetualBooster_number_of_trees", x$.ptr, PACKAGE = "perpetual")
  base_score <- .Call("PerpetualBooster_base_score", x$.ptr, PACKAGE = "perpetual")

  cat("Perpetual Boosting Machine model\n")
  cat("Number of trees: ", n_trees, "\n")
  cat("Base score: ", base_score, "\n")
  classes <- attr(x, "classes")
  if (!is.null(classes) && length(classes) > 0) {
    cat("Classes: ", paste(classes, collapse = ", "), "\n")
  }
  invisible(x)
}

#' Save a PerpetualBooster model
#'
#' @param model A \code{PerpetualBooster} object.
#' @param path String, path to save the model.
#'
#' @export
perpetual_save <- function(model, path) {
  if (!inherits(model, "PerpetualBooster")) {
    stop("model must be a PerpetualBooster object")
  }
  .Call("PerpetualBooster_save_booster", model$.ptr, path, PACKAGE = "perpetual")
}

#' Load a PerpetualBooster model
#'
#' @param path String, path to the saved model.
#'
#' @return A \code{PerpetualBooster} object.
#' @export
perpetual_load <- function(path) {
  if (!file.exists(path)) {
    stop("File not found: ", path)
  }
  
  ptr <- .Call("PerpetualBooster_load_booster", path, PACKAGE = "perpetual")
  model <- structure(list(.ptr = ptr), class = "PerpetualBooster")
  
  classes <- .Call("PerpetualBooster_get_classes", model$.ptr, PACKAGE = "perpetual")
  if (length(classes) > 0) {
    attr(model, "classes") <- classes
  }
  
  objective <- .Call("PerpetualBooster_get_objective", model$.ptr, PACKAGE = "perpetual")
  if (!is.null(objective) && nzchar(objective)) {
    attr(model, "objective") <- objective
  }
  
  return(model)
}

#' Calibrate a PerpetualBooster model
#'
#' @param model A \code{PerpetualBooster} object.
#' @param x Validation features.
#' @param y Validation targets.
#' @param x_cal Calibration features.
#' @param y_cal Calibration targets.
#' @param alpha Calibration parameter.
#'
#' @export
perpetual_calibrate <- function(model, x, y, x_cal, y_cal, alpha) {
  if (is.data.frame(x)) x <- as.matrix(x)
  if (is.data.frame(x_cal)) x_cal <- as.matrix(x_cal)
  storage.mode(x) <- "double"
  storage.mode(x_cal) <- "double"
  
  .Call("PerpetualBooster_calibrate", model$.ptr, 
        as.vector(x), as.integer(nrow(x)), as.integer(ncol(x)), as.numeric(y),
        as.vector(x_cal), as.integer(nrow(x_cal)), as.integer(ncol(x_cal)), as.numeric(y_cal),
        as.numeric(alpha), PACKAGE = "perpetual")
  invisible(model)
}

#' Get feature importance
#'
#' @param model A \code{PerpetualBooster} object.
#' @param method String, method for importance (e.g. "gain").
#' @param normalize Boolean.
#'
#' @return A named vector of importances.
#' @export
perpetual_importance <- function(model, method = "gain", normalize = TRUE) {
  imp <- .Call("PerpetualBooster_calculate_feature_importance", model$.ptr, method, normalize, PACKAGE = "perpetual")
  # imp is a list (from Rust) but with names. Wait, in Rust we returned a Named Vector!
  # So imp is a named numeric vector
  return(imp)
}

# Compatibility for old R6 style usage (optional, but good for backward compat)
PerpetualBooster <- list(
    new = function(...) {
        stop("Please use perpetual() instead of PerpetualBooster$new()")
    },
    load_booster = function(path) {
        perpetual_load(path)
    }
)
