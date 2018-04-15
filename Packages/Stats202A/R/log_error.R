#' Logistic Error
#'
#' This function calculates Error of logistic regression
#' @param X an n x p matrix of explanatory variables
#' @param my_coef reponse of logistic regression
#' @keywords Logistic Regression LR error
#' @export

log_error <- function(X, my_coef){

  # Find the standard error of Logistic Regression

  X_beta_product = X %*% my_coef
  pr = expit(X_beta_product)
  w = diag(c(pr*(1-pr)))

  std_error = solve((t(X) %*% w) %*% X)

  return(sqrt(diag(std_error)))


}
