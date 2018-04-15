#' Priciple Composition analysis
#'
#'  Perform PCA on matrix A using your QR function.
#' @param A  Square matrix
#' @keywords QR decomposition PCA
#' @export


myPCA <- function(A, numIter = 1000){
  
  ## Perform PCA on matrix A using your QR function, myQRC.
  ## Input:
  ## A: Square matrix
  ## numIter: Number of iterations
  
  ########################
  ## FILL IN CODE BELOW ##
  ######################## 
  r = dim(A)[1]
  c = dim(A)[2]
  
  V = matrix(runif(r*r), nrow = r)
  
  for (i in 1 : numIter){
    op = myQR(V)
    Q = op$Q
    V  = A %*% Q
  }
  
  op = myQR(V)
  
  Q = op$Q
  R = op$R
  ## Function should output a list with D and V
  ## D is a vector of eigenvalues of A
  ## V is the matrix of eigenvectors of A (in the 
  ## same order as the eigenvalues in D.)
  return(list("D" = diag(R), "V" = Q))
  
}