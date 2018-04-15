#########################################################
## Stat 202A - Homework 3
## Author: Shraddha Manchekar
## Date : 10/26/2017
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your 
## work since R will attempt to change my working directory
## to one that does not exist.
#############################################################

##################################
## Function 1: QR decomposition ##
##################################

myQR <- function(A){
  
  ## Perform QR decomposition on the matrix A
  ## Input: 
  ## A, an n x m matrix
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  
  n <- dim(A)[1]
  m <- dim(A)[2]
  R <- A
  Q <- diag(n)
  
  for(k in 1:(m-1))
  {
    X <- matrix(0, n, 1)
    X[k:n, 1] <- R[k:n, k]
    V <- X
    V[k] <- X[k] + sign(X[k, 1]) * norm(X, type="F")
    S <- norm(V, type = "F")
    
    if(S!=0)
    {
      u <- V/S
      
      R <- R - (2 * (u %*% (t(u) %*% R)))
      Q <- Q - (2 * (u %*% (t(u) %*% Q))) 
    }
    
  }
  
  ## Function should output a list with Q.transpose and R
  ## Q is an orthogonal n x n matrix
  ## R is an upper triangular n x m matrix  
  ## Q and R satisfy the equation: A = Q %*% R
  return(list("Q" = t(Q), "R" = R))
  
}

###############################################
## Function 2: Linear regression based on QR ##
###############################################

myLM <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Do NOT simulate data in this function. n and p
  ## should be determined by X.
  ## Use myQR inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  z <- cbind(matrix(1, n, 1), X, matrix(Y, n, 1))
  #a <- t(z) %*% z
  b <- myQR(z)
  
  R <- b$R
  
  R1 <- R[1:(p+1), 1:(p+1)]
  Y1 <- R[1: (p+1), p+2]
  
  beta_ls = solve(R1, Y1)
  
  
  ## Function returns the 1 x (p + 1) vector beta_ls, 
  ## the least squares solution vector
  return(beta_ls)
  
}

##################################
## Function 3: PCA based on QR  ##
##################################

myEigen_QR <- function(A, numIter = 1000){
  
  ## Perform PCA on matrix A using your QR function, myQRC.
  ## Input:
  ## A: Square matrix
  ## numIter: Number of iterations
  
  ########################
  ## FILL IN CODE BELOW ##
  ######################## 
  
  r <- dim(A)[1]
  c <- dim(A)[2]
  
  V <- matrix(rnorm(r*r), nrow=r)
  for(i in 1:numIter)
  {
    tmp <- myQR(V)
    Q <- tmp$Q
    V <- A %*% Q
  }
  
  t <- myQR(V)
  Q <- t$Q
  R <- t$R
  
  ## Function should output a list with D and V
  ## D is a vector of eigenvalues of A
  ## V is the matrix of eigenvectors of A (in the 
  ## same order as the eigenvalues in D.)
  return(list("D" = diag(R), "V" = Q))
}

# #######################################################
# # Optional examples (comment out before submitting!) ##
# #######################################################
# 
#   testing_Linear_Regression <- function(){
# 
#     ## This function is not graded; you can use it to
#     ## test out the 'myLinearRegression' function
# 
#     ## Define parameters
#     n    <- 100
#     p    <- 3
# 
#     ## Simulate data from our assumed model.
#     ## We can assume that the true intercept is 0
#     X    <- matrix(rnorm(n * p), nrow = n)
#     #beta <- matrix(1:p, nrow = p)
#     #Y    <- X %*% beta + rnorm(n)
# 
#     ## Save R's linear regression coefficients
#     #R_coef  <- coef(lm(Y ~ X))
#     R_coef  <- eigen(t(X) %*% X)
# 
#     ## Save our linear regression coefficients
#     my_coef <- myEigen_QRC(t(X) %*% X )
# 
#     print(R_coef)
#     print(my_coef)
# 
#     #Are these two vectors different?
#     # sum_square_diff <- sum((R_coef - my_coef)^2)
#     # if(sum_square_diff <= 0.001){
#     #   print('Both results are identical')
#     # }else{
#     #   print('There seems to be a problem...')
#     # }
# 
#   }
# 
#  testing_Linear_Regression()
