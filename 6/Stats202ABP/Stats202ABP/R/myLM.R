#' Linear Regression
#'
#' This function does Linear Regression.
#' @param X an n x p matrix of explanatory variables
#' @param Y is an n dimensional vector of responses
#' @keywords Linear Regression LR
#' @export

myLM <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Use myQR (or myQRC) inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  n <- nrow(X)
  p <- ncol(X)
  
  ## Stack (X, Y) and solve it by QR decomposition
  Z <- cbind(rep(1,n), X, Y)
  R <- myQR(Z)$R
  
  R1 <- R[1:(p + 1), 1:(p + 1)]
  Y1 <- R[1:(p + 1), p + 2]
  
  beta_ls <- solve(R1, Y1)


  error = LM_error(X, Y, beta_ls)

  #beta_ls <- beta_ls[2 : (p + 1)]
  ## Function returns the least squares solution vector
  return(list("coefficients" = beta_ls, "standard_error" = error))
  
}