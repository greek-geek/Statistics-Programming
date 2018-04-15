
#' Logistic Regression
#'
#' Perform the logistic regression of Y on X
#' @param X an n x p matrix of explanatory variables
#' @param Y is an n dimensional vector of responses
#' @keywords Logistic Regression
#' @export     


## Expit/sigmoid function


myLogistic <- function(X, Y){

  ## Perform the logistic regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of binary responses
  ## Use myLM (or myLMC) inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################
  
  rows <- nrow(X)
  cols <- ncol(X)
  beta <- matrix(rep(0, cols), nrow = cols)
  epsilon <- 1e-6
  repeat{
    eta <- X %*% beta
    pr <- expit(eta)
    w <- pr * (1 - pr)
    z <- eta + (Y - pr) / w
    sw <- sqrt(w)
    mw <- matrix(sw, rows, cols)
    xWork <- mw * X
    yWork <- sw * z
    
    betaNew <- myLM(xWork, yWork)$coefficients
    betaNew <- betaNew[2: (p+1)]
    error <- sum(abs(betaNew - beta))
    beta <- betaNew
    if (error < epsilon)
      break
  }
  
  error = log_error(X, beta)
  ## Function returns the logistic regression solution vector
  return(list("coefficients" = beta, "standard_error" = error))
    
}


# Optional testing (comment out!)
# n <- 100
# p <- 4
# 
# X    <- matrix(rnorm(n * p), nrow = n)
# beta <- rnorm(p)
# Y    <- 1 * (runif(n) < expit(X %*% beta))
# 
# logistic_beta <- myLogistic(X, Y)
# print(logistic_beta)
# print(glm(formula = Y ~ X + 0, family=binomial("logit")))

