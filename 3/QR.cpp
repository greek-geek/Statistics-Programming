/*
#########################################################
## Stat 202A - Homework 3
## Author: 
## Date : 
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################
 
###########################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not change your working directory
## anywhere inside of your code. If you do, I will be unable 
## to grade your work since R will attempt to change my 
## working directory to one that does not exist.
###########################################################
 
*/ 


# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;


/* ~~~~~~~~~~~~~~~~~~~~~~~~~ 
 Sign function for later use 
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
double signC(double d){
  return d<0?-1:d>0? 1:0;
}



/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
   Problem 1: QR decomposition 
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */  
  

// [[Rcpp::export()]]
List myQRC(const mat A){ 
  
  /*
  Perform QR decomposition on the matrix A
  Input: 
  A, an n x m matrix (mat)

  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
  
  */ 
  int n = A.n_rows;
  int m = A.n_cols;
  mat R = mat(A);
  mat Q, X, V;
  Q = eye<mat>(n, n);
  
  for(int k=0; k<(m-1); k++)
  {
    X = zeros<mat>(n, 1);
    
    for(int l=k;l<n;l++){
      X(l,0) = R(l,k);
    }
    
    V = X;
    V(k) = X(k) + signC(X(k, 0)) * norm(X, "fro");
    double S = norm(V, "fro");
    
    if(S!=0)
    {
      mat u = V/S;
      R = R - 2 * (u * (u.t() * R));
      Q = Q - 2 * (u * (u.t() * Q));
    }
    
  }
  
  
  
  List output;
  // Function should output a List 'output', with 
  // Q.transpose and R
  // Q is an orthogonal n x n matrix
  // R is an upper triangular n x m matrix
  // Q and R satisfy the equation: A = Q %*% R
  output["Q"] = Q.t();
  output["R"] = R;
  return(output);
  

}
  
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
   Problem 2: Linear regression using QR 
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  
  
// [[Rcpp::export()]]
mat myLinearRegressionC(const mat X, const mat Y){
    
  /*  
  Perform the linear regression of Y on X
  Input: 
  X is an n x p matrix of explanatory variables
  Y is an n dimensional vector of responses
  Do NOT simulate data in this function. n and p
  should be determined by X.
  Use myQRC inside of this function
  
  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
  
  */  
  
  int n = X.n_rows;
  int p = X.n_cols;
  mat z(n, 1+p+Y.n_cols);
  
  z = join_rows(ones<mat>(1, n).t(), X);
  z = join_rows(z, Y);
  
  List b = myQRC(z);
  
  mat R = b["R"];
  
  mat R1 = R.submat(0, 0, p, p);
  mat Y1 = R.submat(0, p+1, p, p+1);
  
  mat beta_ls = solve(R1, Y1);
  // Function returns the 'p+1' by '1' matrix 
  // beta_ls of regression coefficient estimates
  return(beta_ls.t());
  
}  

/* ~~~~~~~~~~~~~~~~~~~~~~~~ 
 Problem 3: PCA based on QR 
 ~~~~~~~~~~~~~~~~~~~~~~~~~~ */


// [[Rcpp::export()]]
List myEigen_QRC(const mat A, const int numIter = 1000){
  
  /*  
  
  Perform PCA on matrix A using your QR function, myQRC.
  Input:
  A: Square matrix
  numIter: Number of iterations
   
  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
   
   */  
  
  int r = A.n_rows;
  //int c = A.n_cols;
  List tmp;
  mat V = randu<mat>(r, r);
  for(int i=0; i<numIter; i++)
  {
    tmp = myQRC(V);
    mat Q = tmp["Q"];
    V = A*Q;
  }
  tmp = myQRC(V);
  mat Q = tmp["Q"];
  mat R = tmp["R"];
  
  mat D = R.diag();
  D.reshape(1, D.n_rows);
  List output;
  // Function should output a list with D and V
  // D is a vector of eigenvalues of A
  // V is the matrix of eigenvectors of A (in the 
  // same order as the eigenvalues in D.)
  output["D"] = D;
  output["V"] = Q;
  return(output);

};