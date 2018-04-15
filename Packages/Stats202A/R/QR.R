#' QR Function
#'
#' This function performs QR decomposition on the matrix A
#' @param A an n x m matrix
#' @keywords QR decomposition
#' @export

QR <- function(A){

  ## Performs QR decomposition on the matrix A
  ## Input:
  ## A, an n x m matrix
  ## Function should output a list with Q.transpose and R
  ## Q is an orthogonal n x n matrix
  ## R is an upper triangular n x m matrix
  ## Q and R satisfy the equation: A = Q %*% R


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

  return(list("Q" = t(Q), "R" = R))

}
