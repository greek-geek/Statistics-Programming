#' Logistic Regression Function
#'
#' This function performs logistic regression of Y on X.
#' @param X n x p matrix of explanatory variables
#' @param Y n dimensional vector of binary responses
#' @keywords Logistic Regression
#' @export


LogisticRegression <- function(X, Y){

  ## Performs logistic regression of Y on X
  ## Input:
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of binary responses
  ## Function returns the logistic regression solution vector

  n = dim(X)[1]
  p = dim(X)[2]

  beta = matrix(rep(0,p), nrow = p)

  epsilon = 1e-6

  repeat
  {
    ETA = X %*% beta
    pr = expit(ETA)
    z = ETA + (Y - pr)/(pr*(1-pr))
    sqw = sqrt(pr*(1-pr))
    mw = matrix(sqw, n, p)
    xw = mw*X
    yw = sqw*z

    beta_n = LinearRegression(xw, yw)$coefficients
    beta_n = beta_n[2: (p+1)]
    error = sum(abs(beta_n - beta))
    beta = beta_n
    if (error < epsilon)
      break
  }
  error = log_error(X, beta)

  return(list("coefficients" = beta, "standard_error" = error))

}
