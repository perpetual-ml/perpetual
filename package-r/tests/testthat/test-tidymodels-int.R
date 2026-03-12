# ============================================================
# tests/testthat/test-perpsnip.R
# ============================================================

skip_if_not_installed("parsnip")
skip_if_not_installed("workflows")
skip_if_not_installed("dplyr")

# ============================================================
# helpers
# ============================================================

cls_data <- function() {
  dplyr::mutate(mtcars, am = factor(as.character(am), levels = c("0", "1")))
}

reg_data <- function() {
  mtcars
}

# ============================================================
# model spec construction
# ============================================================

test_that("perpsnip() returns a model_spec of class perpsnip", {
  spec <- perpsnip()

  expect_s3_class(spec, "perpsnip")
  expect_s3_class(spec, "model_spec")
})

test_that("perpsnip() defaults to classification mode", {
  spec <- perpsnip()

  expect_equal(spec$mode, "classification")
})

test_that("perpsnip() defaults to perpetual engine", {
  spec <- perpsnip()

  expect_equal(spec$engine, "perpetual")
})

test_that("perpsnip() accepts all valid modes", {
  expect_no_error(perpsnip(mode = "classification"))
  expect_no_error(perpsnip(mode = "regression"))
  expect_no_error(perpsnip(mode = "censored regression"))
})

test_that("perpsnip() rejects invalid mode", {
  expect_error(perpsnip(mode = "unknown"))
})

test_that("perpsnip() stores parameters correctly", {
  spec <- perpsnip(budget = 0.5, seed = 42, num_threads = 2)

  expect_equal(rlang::quo_get_expr(spec$args$budget), 0.5)
  expect_equal(rlang::quo_get_expr(spec$args$seed), 42L)
  expect_equal(rlang::quo_get_expr(spec$args$num_threads), 2L)
})

# ============================================================
# update method
# ============================================================

test_that("update.perpsnip() updates parameters correctly", {
  spec <- perpsnip(budget = 0.5)
  updated <- update.perpsnip(spec, budget = 1.0)

  expect_equal(rlang::quo_get_expr(updated$args$budget), 1.0)
})

test_that("update.perpsnip() with fresh = TRUE resets unspecified args to NULL", {
  spec <- perpsnip(budget = 0.5, seed = 42)
  updated <- update.perpsnip(spec, budget = 1.0, fresh = TRUE)

  expect_equal(rlang::quo_get_expr(updated$args$budget), 1.0)
  expect_null(rlang::quo_get_expr(updated$args$seed))
})

test_that("update.perpsnip() preserves unspecified args when fresh = FALSE", {
  spec <- perpsnip(budget = 0.5, seed = 42)
  updated <- update.perpsnip(spec, budget = 1.0)

  expect_equal(rlang::quo_get_expr(updated$args$seed), 42L)
})

# ============================================================
# fit + predict: classification
# ============================================================

test_that("perpsnip() fits and predicts class for binary classification", {
  wf <- workflows::workflow() |>
    workflows::add_formula(am ~ mpg + cyl + disp + hp + wt) |>
    workflows::add_model(perpsnip(mode = "classification", budget = 0.5)) |>
    generics::fit(data = cls_data())

  preds <- predict(wf, new_data = cls_data())

  expect_s3_class(preds, "tbl_df")
  expect_named(preds, ".pred_class")
  expect_equal(nrow(preds), nrow(mtcars))
})

test_that("perpsnip() fits and predicts prob for binary classification", {
  wf <- workflows::workflow() |>
    workflows::add_formula(am ~ mpg + cyl + disp + hp + wt) |>
    workflows::add_model(perpsnip(mode = "classification", budget = 0.5)) |>
    generics::fit(data = cls_data())

  preds <- predict(wf, new_data = cls_data(), type = "prob")

  expect_s3_class(preds, "tbl_df")
  expect_equal(ncol(preds), 2L)
  expect_equal(nrow(preds), nrow(mtcars))
  expect_true(all(rowSums(dplyr::select(preds, where(is.numeric))) > 0.99 &
    rowSums(dplyr::select(preds, where(is.numeric))) < 1.01))
})

test_that("perpsnip() prob columns are named after factor levels", {
  wf <- workflows::workflow() |>
    workflows::add_formula(am ~ mpg + cyl + disp + hp + wt) |>
    workflows::add_model(perpsnip(mode = "classification", budget = 0.5)) |>
    generics::fit(data = cls_data())

  preds <- predict(wf, new_data = cls_data(), type = "prob")

  expect_named(preds, c(".pred_0", ".pred_1"))
})

# ============================================================
# fit + predict: regression
# ============================================================

test_that("perpsnip() fits and predicts for regression", {
  wf <- workflows::workflow() |>
    workflows::add_formula(mpg ~ cyl + disp + hp + wt) |>
    workflows::add_model(perpsnip(mode = "regression", budget = 0.5)) |>
    generics::fit(data = reg_data())

  preds <- predict(wf, new_data = reg_data())

  expect_s3_class(preds, "tbl_df")
  expect_named(preds, ".pred")
  expect_equal(nrow(preds), nrow(mtcars))
  expect_true(is.numeric(preds$.pred))
})

test_that("perpsnip() regression predictions are finite", {
  wf <- workflows::workflow() |>
    workflows::add_formula(mpg ~ cyl + disp + hp + wt) |>
    workflows::add_model(perpsnip(mode = "regression", budget = 0.5)) |>
    generics::fit(data = reg_data())

  preds <- predict(wf, new_data = reg_data())

  expect_true(all(is.finite(preds$.pred)))
})
