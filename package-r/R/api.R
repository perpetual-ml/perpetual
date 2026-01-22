#' Train a PerpetualBooster model
#'
#' @param x A matrix or data frame of features.
#' @param y A numeric vector of targets.
#' @param objective Objective function (e.g. "LogLoss", "SquaredLoss").
#' @param budget Training budget.
#' @param iteration_limit Limit on number of iterations.
#' @param stopping_rounds Number of rounds for early stopping.
#' @param max_bin Maximum number of bins.
#' @param num_threads Number of threads to use.
#' @param missing Value to consider missing.
#' @param allow_missing_splits Should missing splits be allowed?
#' @param create_missing_branch Should a separate branch be created for missing?
#' @param missing_node_treatment Treatment for missing nodes.
#' @param log_iterations Log iterations.
#' @param quantile Quantile for quantile regression.
#' @param reset Reset the model.
#' @param timeout Timeout in seconds.
#' @param memory_limit Memory limit in GB.
#' @param seed Random seed.
#' @param ... Additional parameters for PerpetualBooster$new.
#' @return A fitted PerpetualBooster object with classes attribute.
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
  
  model <- PerpetualBooster$new(objective = objective, budget = budget, 
                                iteration_limit = iteration_limit, 
                                stopping_rounds = stopping_rounds, 
                                max_bin = max_bin, num_threads = num_threads,
                                missing = missing, allow_missing_splits = allow_missing_splits,
                                create_missing_branch = create_missing_branch,
                                missing_node_treatment = missing_node_treatment,
                                log_iterations = log_iterations,
                                quantile = quantile, reset = reset,
                                timeout = timeout, memory_limit = memory_limit,
                                seed = seed, ...)
  model$fit(flat_data, rows, cols, y)
  
  # Store classes and objective on the R object for predict()
  attr(model, "classes") <- classes
  attr(model, "objective") <- objective
  
  return(model)
}

#' Predict using a PerpetualBooster model
#'
#' @param object A PerpetualBooster object.
#' @param newdata A matrix or data frame of new data.
#' @param type Type of prediction: "class" (label), "prob" (probability), "raw" (log-odds/raw score),
#' "contribution" (SHAP values), or "interval" (prediction intervals).
#' @param method Contribution method (default "average"). Only used if type = "contribution".
#' @param ... Not used.
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
  
  # Get raw predictions
  raw_preds <- .Call("wrap__PerpetualBooster__predict", object$.ptr, flat_data, as.integer(rows), as.integer(cols), PACKAGE = "perpetual")
  
  # Get classes from attribute (set during training)
  classes <- attr(object, "classes")
  objective <- attr(object, "objective")
  
  if (type == "raw") {
    return(raw_preds)
  }
  
  if (type == "contribution") {
    # Method can be one of: "weight", "average", "branchdifference", "midpointdifference", "modedifference", "probabilitychange", "shapley"
    contribs <- .Call("wrap__PerpetualBooster__predict_contributions", object$.ptr, flat_data, as.integer(rows), as.integer(cols), method, PACKAGE = "perpetual")
    
    # Reshape contribs (rows x (cols + 1))
    contrib_mat <- matrix(contribs, nrow = rows, ncol = cols + 1, byrow = FALSE)
    return(contrib_mat)
  }
  
  if (type == "interval") {
    intervals <- .Call("wrap__PerpetualBooster__predict_intervals", object$.ptr, flat_data, as.integer(rows), as.integer(cols), PACKAGE = "perpetual")
    return(intervals)
  }
  
  # If classes not available (loaded model), try to infer from prediction shape
  if (is.null(classes) || length(classes) == 0) {
    # Assume regression or simple binary
    if (type == "prob") {
      # Apply sigmoid for binary classification
      probs <- 1 / (1 + exp(-raw_preds))
      return(probs)
    } else {
      return(raw_preds)
    }
  }
  
  if (length(classes) <= 2) {
    # Binary classification or Regression
    if (length(classes) == 2 && !is.null(objective) && objective == "LogLoss") {
      # Binary classification
      probs <- 1 / (1 + exp(-raw_preds))
      if (type == "prob") {
        return(probs)
      } else {
        return(ifelse(probs >= 0.5, classes[2], classes[1]))
      }
    } else {
      # Regression or single class
      return(raw_preds)
    }
  } else {
    # Multiclass
    n_classes <- length(classes)
    # Reshape raw_preds (column-major from Rust flat_map)
    preds_mat <- matrix(raw_preds, nrow = rows, ncol = n_classes, byrow = FALSE)
    
    if (type == "prob") {
      # Use Rust side predict_proba if available
      probs_flat <- .Call("wrap__PerpetualBooster__predict_proba", object$.ptr, flat_data, as.integer(rows), as.integer(cols), PACKAGE = "perpetual")
      probs <- matrix(probs_flat, nrow = rows, ncol = n_classes, byrow = FALSE)
      # Normalize so each row sums to 1 (One-vs-Rest doesn't guarantee this)
      row_sums <- rowSums(probs)
      probs <- probs / row_sums
      return(probs)
    } else {
      # Class labels
      # Still use raw_preds for class labels to avoid double softmax if not needed,
      # but we need to reshape correctly.
      preds_mat <- matrix(raw_preds, nrow = rows, ncol = n_classes, byrow = FALSE)
      max_idx <- max.col(preds_mat, ties.method = "first")
      return(classes[max_idx])
    }
  }
}

#' Print PerpetualBooster
#' @export
print.PerpetualBooster <- function(x, ...) {
  cat("Perpetual Boosting Machine model\n")
  cat("Number of trees: ", x$number_of_trees(), "\n")
  cat("Base score: ", x$base_score(), "\n")
  classes <- attr(x, "classes")
  if (!is.null(classes) && length(classes) > 0) {
    cat("Classes: ", paste(classes, collapse = ", "), "\n")
  }
  invisible(x)
}

#' Save a PerpetualBooster model
#'
#' @param model A PerpetualBooster object.
#' @param path Path to save the model (as JSON).
#' @export
perpetual_save <- function(model, path) {
  if (!inherits(model, "PerpetualBooster")) {
    stop("model must be a PerpetualBooster object")
  }
  model$save_booster(path)
}

#' Load a PerpetualBooster model
#'
#' @param path Path to the saved model (JSON).
#' @return A PerpetualBooster object.
#' @export
perpetual_load <- function(path) {
  if (!file.exists(path)) {
    stop("File not found: ", path)
  }
  model <- PerpetualBooster$load_booster(path)
  
  # Try to set attributes from metadata if available (standard rextendr load might not preserve them)
  # But the Rust side load_booster already populates internal fields.
  # If we need classes here, we can extract them if we expose a get_classes method.
  classes <- .Call("wrap__rust_get_classes", model$.ptr, PACKAGE = "perpetual")
  if (length(classes) > 0) {
    attr(model, "classes") <- classes
  }
  
  # Restore objective
  objective <- .Call("wrap__rust_get_objective", model$.ptr, PACKAGE = "perpetual")
  if (!is.null(objective) && nzchar(objective)) {
    attr(model, "objective") <- objective
  }
  
  return(model)
}

#' Calibrate a PerpetualBooster model for prediction intervals
#'
#' @param model A PerpetualBooster object.
#' @param x Training data.
#' @param y Training targets.
#' @param x_cal Calibration data.
#' @param y_cal Calibration targets.
#' @param alpha Alpha values (1 - coverage).
#' @export
perpetual_calibrate <- function(model, x, y, x_cal, y_cal, alpha) {
  if (is.data.frame(x)) x <- as.matrix(x)
  if (is.data.frame(x_cal)) x_cal <- as.matrix(x_cal)
  storage.mode(x) <- "double"
  storage.mode(x_cal) <- "double"
  
  .Call("wrap__PerpetualBooster__calibrate", model$.ptr, 
        as.vector(x), as.integer(nrow(x)), as.integer(ncol(x)), as.numeric(y),
        as.vector(x_cal), as.integer(nrow(x_cal)), as.integer(ncol(x_cal)), as.numeric(y_cal),
        as.numeric(alpha), PACKAGE = "perpetual")
  invisible(model)
}

#' Get feature importance from a PerpetualBooster model
#'
#' @param model A PerpetualBooster object.
#' @param method Importance method ("weight", "gain", "totalgain", "cover", "totalcover").
#' @param normalize Whether to normalize importance values.
#' @return A named numeric vector of feature importance.
#' @export
perpetual_importance <- function(model, method = "gain", normalize = TRUE) {
  imp <- .Call("wrap__PerpetualBooster__calculate_feature_importance", model$.ptr, method, normalize, PACKAGE = "perpetual")
  # imp is a list with names as feature indices
  vals <- unlist(imp)
  names(vals) <- names(imp)
  return(vals)
}
