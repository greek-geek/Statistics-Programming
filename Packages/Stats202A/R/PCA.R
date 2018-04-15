#' PCA Function
#'
#' This function performs PCA on matrix A using QR function, QR.
#' @param A square matrix
#' @param numIter number of iterations
#' @keywords PCA eigen
#' @export


PCA <- function(A, numIter = 1000){

  ## Performs PCA on matrix A using QR function, QR
  ## Input:
  ## A: Square matrix
  ## numIter: Number of iterations
  ## Function outputs a list with D and V
  ## D is a vector of eigenvalues of A
  ## V is the matrix of eigenvectors of A (in the
  ## same order as the eigenvalues in D.)


  r <- dim(A)[1]
  c <- dim(A)[2]

  V <- matrix(rnorm(r*r), nrow=r)
  for(i in 1:numIter)
  {
    tmp <- QR(V)
    Q <- tmp$Q
    V <- A %*% Q
  }

  t <- QR(V)
  Q <- t$Q
  R <- t$R

  return(list("D" = diag(R), "V" = Q))
}
