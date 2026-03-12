#' PerpetualBooster via parsnip
#'
#' A parsnip-compatible model spec for the PerpetualBooster algorithm.
#' Supports classification, regression, and censored regression modes.
#'
#' @param mode A string for the model mode. One of "classification",
#'   "regression", or "censored regression".
#' @param engine A string for the engine. Currently only "perpetual".
#' @inheritParams perpetual
#'
#' @return A \code{model_spec} object of class \code{perpsnip}.
#'
#' @examples
#' \dontrun{
#' library(perpsnip)
#' library(workflows)
#'
#' spec <- perpsnip(mode = "classification", budget = 0.5)
#'
#' wf <- workflow() |>
#'   add_formula(am ~ .) |>
#'   add_model(spec) |>
#'   fit(data = dplyr::mutate(mtcars, am = as.factor(am)))
#' }
#'
#' @export
perpsnip <- function(
    mode = "classification",
    engine = "perpetual",
    objective = NULL,
    budget = NULL,
    iteration_limit = NULL,
    stopping_rounds = NULL,
    max_bin = NULL,
    num_threads = NULL,
    missing = NULL,
    allow_missing_splits = NULL,
    create_missing_branch = NULL,
    missing_node_treatment = NULL,
    log_iterations = NULL,
    quantile = NULL,
    reset = NULL,
    timeout = NULL,
    memory_limit = NULL,
    seed = NULL,
    calibration_method = NULL,
    save_node_stats = NULL
) {
  mode <- rlang::arg_match(
    mode,
    c("classification", "regression", "censored regression")
  )

  args <- list(
    objective = rlang::enquo(objective),
    budget = rlang::enquo(budget),
    iteration_limit = rlang::enquo(iteration_limit),
    stopping_rounds = rlang::enquo(stopping_rounds),
    max_bin = rlang::enquo(max_bin),
    num_threads = rlang::enquo(num_threads),
    missing = rlang::enquo(missing),
    allow_missing_splits = rlang::enquo(allow_missing_splits),
    create_missing_branch = rlang::enquo(create_missing_branch),
    missing_node_treatment = rlang::enquo(missing_node_treatment),
    log_iterations = rlang::enquo(log_iterations),
    quantile = rlang::enquo(quantile),
    reset = rlang::enquo(reset),
    timeout = rlang::enquo(timeout),
    memory_limit = rlang::enquo(memory_limit),
    seed = rlang::enquo(seed),
    calibration_method = rlang::enquo(calibration_method),
    save_node_stats = rlang::enquo(save_node_stats)
  )

  parsnip::new_model_spec(
    cls = "perpsnip",
    args = args,
    eng_args = NULL,
    mode = mode,
    user_specified_mode = !missing(mode),
    method = NULL,
    engine = engine,
    user_specified_engine = !missing(engine)
  )
}


#' @keywords internal
#' @export
print.perpsnip <- function(x, ...) {
  cat("Perpetual Boosting Model Specification (", x$mode, ")\n\n", sep = "")
  parsnip::model_printer(x, ...)
  invisible(x)
}

#' @importFrom stats update
#' @keywords internal
#' @export
update.perpsnip <- function(
    object,
    parameters = NULL,
    objective = NULL,
    budget = NULL,
    iteration_limit = NULL,
    stopping_rounds = NULL,
    max_bin = NULL,
    num_threads = NULL,
    missing = NULL,
    allow_missing_splits = NULL,
    create_missing_branch = NULL,
    missing_node_treatment = NULL,
    log_iterations = NULL,
    quantile = NULL,
    reset = NULL,
    timeout = NULL,
    memory_limit = NULL,
    seed = NULL,
    calibration_method = NULL,
    save_node_stats = NULL,
    fresh = FALSE,
    ...
) {
  args <- list(
    objective = rlang::enquo(objective),
    budget = rlang::enquo(budget),
    iteration_limit = rlang::enquo(iteration_limit),
    stopping_rounds = rlang::enquo(stopping_rounds),
    max_bin = rlang::enquo(max_bin),
    num_threads = rlang::enquo(num_threads),
    missing = rlang::enquo(missing),
    allow_missing_splits = rlang::enquo(allow_missing_splits),
    create_missing_branch = rlang::enquo(create_missing_branch),
    missing_node_treatment = rlang::enquo(missing_node_treatment),
    log_iterations = rlang::enquo(log_iterations),
    quantile = rlang::enquo(quantile),
    reset = rlang::enquo(reset),
    timeout = rlang::enquo(timeout),
    memory_limit = rlang::enquo(memory_limit),
    seed = rlang::enquo(seed),
    calibration_method = rlang::enquo(calibration_method),
    save_node_stats = rlang::enquo(save_node_stats)
  )

  parsnip::update_spec(
    object = object,
    parameters = parameters,
    args_enquo_list = args,
    fresh = fresh,
    cls = "perpsnip"
  )
}

#' @keywords internal
#' @noRd
perpetual_prob_convert <- function(out, object) {
  classes <- attr(object$fit, "classes")

  if (is.null(classes) || length(classes) == 0) {
    classes <- c("0", "1")
    out <- data.frame(`0` = 1 - out, `1` = out, check.names = FALSE)
  } else if (is.matrix(out)) {
    out <- as.data.frame(out)
    colnames(out) <- as.character(classes)
  } else {
    cls <- as.character(classes)
    out <- data.frame(a = 1 - out, b = out, check.names = FALSE)
    colnames(out) <- cls
  }

  tibble::as_tibble(out)
}
