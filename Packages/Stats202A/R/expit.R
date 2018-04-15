#' Sigmoid Function
#'
#' This function applies sigmoid function on X
#' @param x a variable
#' @keywords sigmoid expit
#' @export

expit <- function(x){

  ## Expit/sigmoid function

  1 / (1 + exp(-x))
}
