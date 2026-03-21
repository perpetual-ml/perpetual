register_perpsnip <- function() {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    return(invisible(NULL))
  }

  model_env <- parsnip::get_model_env()

  if ("perpsnip" %in% model_env$models) {
    if (interactive() || identical(Sys.getenv("DEVTOOLS_LOAD"), "true")) {
      if (exists("perpsnip", envir = model_env)) {
        rm(list = "perpsnip", envir = model_env)
      }
      model_env$models <- setdiff(model_env$models, "perpsnip")
    } else {
      return(invisible(TRUE))
    }
  }

  parsnip::set_new_model("perpsnip")
  parsnip::set_model_mode("perpsnip", "classification")
  parsnip::set_model_mode("perpsnip", "regression")
  parsnip::set_model_mode("perpsnip", "censored regression")

  # --- classification ---
  parsnip::set_model_engine("perpsnip", mode = "classification", eng = "perpetual")
  parsnip::set_dependency("perpsnip", eng = "perpetual", pkg = "perpetual", mode = "classification")

  parsnip::set_fit(
    model = "perpsnip",
    eng = "perpetual",
    mode = "classification",
    value = list(
      interface = "matrix",
      protect = c("x", "y"),
      func = c(pkg = "perpetual", fun = "perpetual_class"),
      defaults = list(objective = "LogLoss")
    )
  )

  parsnip::set_encoding(
    model = "perpsnip",
    eng = "perpetual",
    mode = "classification",
    options = list(
      predictor_indicators = "none",
      compute_intercept = FALSE,
      remove_intercept = FALSE,
      allow_sparse_x = FALSE
    )
  )

  parsnip::set_pred(
    model = "perpsnip",
    eng = "perpetual",
    mode = "classification",
    type = "class",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        newdata = rlang::expr(new_data),
        type = "class"
      )
    )
  )

  parsnip::set_pred(
    model = "perpsnip",
    eng = "perpetual",
    mode = "classification",
    type = "prob",
    value = list(
      pre = NULL,
      post = perpetual_prob_convert,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        newdata = rlang::expr(new_data),
        type = "prob"
      )
    )
  )

  parsnip::set_pred(
    model = "perpsnip",
    eng = "perpetual",
    mode = "classification",
    type = "raw",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        newdata = rlang::expr(new_data),
        type = "raw"
      )
    )
  )

  # --- regression ---
  parsnip::set_model_engine("perpsnip", mode = "regression", eng = "perpetual")
  parsnip::set_dependency("perpsnip", eng = "perpetual", pkg = "perpetual", mode = "regression")

  parsnip::set_fit(
    model = "perpsnip",
    eng = "perpetual",
    mode = "regression",
    value = list(
      interface = "matrix",
      protect = c("x", "y"),
      func = c(pkg = "perpetual", fun = "perpetual"),
      defaults = list(objective = "SquaredLoss")
    )
  )

  parsnip::set_encoding(
    model = "perpsnip",
    eng = "perpetual",
    mode = "regression",
    options = list(
      predictor_indicators = "none",
      compute_intercept = FALSE,
      remove_intercept = FALSE,
      allow_sparse_x = FALSE
    )
  )

  parsnip::set_pred(
    model = "perpsnip",
    eng = "perpetual",
    mode = "regression",
    type = "numeric",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        newdata = rlang::expr(new_data),
        type = "raw"
      )
    )
  )

  # --- censored regression ---
  if (requireNamespace("censored", quietly = TRUE)) {
    parsnip::set_model_engine("perpsnip", mode = "censored regression", eng = "perpetual")
    parsnip::set_dependency("perpsnip", eng = "perpetual", pkg = "perpetual", mode = "censored regression")

    parsnip::set_fit(
      model = "perpsnip",
      eng = "perpetual",
      mode = "censored regression",
      value = list(
        interface = "matrix",
        protect = c("x", "y"),
        func = c(pkg = "perpetual", fun = "perpetual"),
        defaults = list(objective = "SurvivalLogLikelihood")
      )
    )

    parsnip::set_encoding(
      model = "perpsnip",
      eng = "perpetual",
      mode = "censored regression",
      options = list(
        predictor_indicators = "none",
        compute_intercept = FALSE,
        remove_intercept = FALSE,
        allow_sparse_x = FALSE
      )
    )

    parsnip::set_pred(
      model = "perpsnip",
      eng = "perpetual",
      mode = "censored regression",
      type = "numeric",
      value = list(
        pre = NULL,
        post = NULL,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit),
          newdata = rlang::expr(new_data),
          type = "raw"
        )
      )
    )
  }
}
