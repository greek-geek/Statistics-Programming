#' Linear Regression Function
#'
#' This function performs linear regression of Y on X.
#' @param X n x p matrix of explanatory variables
#' @param Y n dimensional vector of responses
#' @keywords Linear Regression
#' @export


LinearRegression <- function(X, Y){

  ## Performs linear regression of Y on X
  ## Input:
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Function returns the 1 x (p + 1) vector beta_ls,
  ## the least squares solution vector

  n <- dim(X)[1]
  p <- dim(X)[2]

  z <- cbind(matrix(1, n, 1), X, matrix(Y, n, 1))
  #a <- t(z) %*% z
  b <- QR(z)

  R <- b$R

  R1 <- R[1:(p+1), 1:(p+1)]
  Y1 <- R[1: (p+1), p+2]

  beta_ls = solve(R1, Y1)
  error = LM_error(X, Y, beta_ls)


  return(list("coefficients" = beta_ls, "standard_error" = error))

}
