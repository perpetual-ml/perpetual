#' @export
pgb.train <- function(x, y) {
  .Call("R_train_univariate", as.integer(x), as.integer(y))
}

#' @export
predict_model <- function(x) {
  .Call(
    "R_predict_univariate",
    x,
    as.integer(1000),
    as.integer(10)
  )
}