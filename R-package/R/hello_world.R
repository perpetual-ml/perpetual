#' @export
foo <- function(x, y) {
  .Call("R_wrapper_univariate", as.integer(x), as.integer(y))
}