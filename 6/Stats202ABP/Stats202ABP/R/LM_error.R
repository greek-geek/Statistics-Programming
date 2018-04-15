#' Linear Regression Error 
#'
#' This function calculates Error of linear regression
#' @param X an n x p matrix of explanatory variables
#' @param Y Output variables
#' @param my_coef reponse of linear regression
#' @keywords Linear Regression LR error
#' @export

LM_error <- function(X, Y, my_coef){
  
  # Find the standard error of Linear Regression
  
  X_bias = cbind(rep(1,n),X)
  XtX_inv = solve(t(X_bias) %*% X_bias)
  
  RSS =  sum(((X_bias %*% matrix(my_coef)) - Y)^2)
  sigma_squared = RSS/(n-p-1)
  
  std_error = sqrt(diag(sigma_squared * XtX_inv))
  
  return(std_error)
  
  
  
}
