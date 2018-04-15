#' expit function
#'
#'sigmoid function
#' @param x is a vector
#' @keywords expit sigmoid
#' @export


expit <- function(x){
  1 / (1 + exp(-x))
}