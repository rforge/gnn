\name{fitGNN}
\alias{fitGNN}
\alias{fitGNN.gnn_GNN}
\alias{fitGNNonce}
\alias{fitGNNonce.gnn_GNN}
\title{Functions for Training of Generative Neural Networks}
\description{
  Functions for training generative neural networks.
}
\usage{
\method{fitGNN}{gnn_GNN}(x, data, batch.size, n.epoch, prior = NULL,
    verbose = 1, ...)
\method{fitGNNonce}{gnn_GNN}(x, data, batch.size, n.epoch, prior = NULL,
    verbose = 1, file = NULL, ...)
}
\arguments{
  \item{x}{
    \describe{
      \item{\code{fitGNN()}}{object of class \code{"gnn_GNN"}
	to be trained.}
      \item{\code{fitGNNonce()}}{object of class \code{"gnn_GNN"}
	to be trained or a list of such.}
    }
  }
  \item{data}{\eqn{(n, d)}-matrix containing the \eqn{n}
    \eqn{d}-dimensional observations of the training data.}
  \item{batch.size}{number of samples used per stochastic gradient step.}
  \item{n.epoch}{number of epochs (one epoch equals one pass through
    the complete training dataset while updating the GNN's parameters
    through stochastic gradient steps).}
  \item{prior}{\eqn{(n, d)}-matrix of prior samples; see also
    \code{\link{rPrior}()}. If \code{prior = NULL} a sample of
    independent N(0,1) random variates is generated.}
  \item{verbose}{see \code{\link{fit.keras.engine.training.Model}()}.}
  \item{file}{\code{NULL} or a \code{\link{character}} string
    specifying the file in which the trained GNN(s) is (are)
    saved. If \code{file} is provided and the specified file exists,
    it is loaded and returned via \code{\link{loadGNN}()}.}
  \item{\dots}{additional arguments passed to the underlying
    \code{fit()} (which is \code{keras:::fit.keras.engine.training.Model()}).}
}
\value{
  The trained GNN (or, in case of \code{fitGNNonce()}, perhaps a
  list of such).
}
\author{Marius Hofert}
\seealso{
  \code{\link{GMMN}()}, \code{\link{saveGNN}()},
  \code{\link{loadGNN}()}.
}
\examples{
\donttest{
## Training data
d <- 2
P <- matrix(0.9, nrow = d, ncol = d)
diag(P) <- 1
A <- t(chol(P))
set.seed(271)
ntrn <- 60000
Z <- matrix(rnorm(ntrn * d), ncol = d)
X <- t(A \%*\% t(Z)) # d-dimensional equicorrelated normal
U <- apply(abs(X), 2, rank) / (ntrn + 1) # pseudo-observations of |X|
plot(U[1:2000,], xlab = expression(U[1]), ylab = expression(U[2]))

## Define the model and 'train' it
GMMN <- GMMN(c(d, 300, d))
GMMN <- fitGNN(GMMN, data = U, batch.size = 500, n.epoch = 2)
## Note: Obviously, in a real-world application, batch.size and n.epoch
##       should be (much) larger (e.g., batch.size = 5000, n.epoch = 300).

## Evaluate (roughly picks up the shape even with our bad choices of
## batch.size and n.epoch)
set.seed(271)
V <- rGNN(GMMN, size = 2000) # sample
plot(V, xlab = expression(V[1]), ylab = expression(V[2]))
}
}
\keyword{optimize}
