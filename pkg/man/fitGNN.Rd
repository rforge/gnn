\name{fitGNN}
\alias{fitGNN}
\alias{fitGNN.gnn_GNN}
\alias{fitGNNonce}
\alias{fitGNNonce.list}
\alias{is.trained}
\alias{is.trained.gnn_GNN}
\alias{is.trained.list}
\title{Functions and Methods for Training of Generative Neural Networks}
\description{
  Functions and methods for training generative neural networks.
}
\usage{
\method{fitGNN}{gnn_GNN}(x, data, batch.size, n.epoch, prior = NULL,
    verbose = 1, ...)
\method{fitGNNonce}{list}(x, data, batch.size, n.epoch, prior = NULL,
    verbose = 1, file = NULL, ...)
\method{is.trained}{gnn_GNN}(x)
\method{is.trained}{list}(x)
}
\arguments{
  \item{x}{
    \describe{
      \item{fitGNN(), is.trained.gnn_GNN()}{object of class
	\code{"gnn_GNN"} to be trained.}
      \item{fitGNNonce(), is.trained.list()}{object of class
	\code{"gnn_GNN"} to be trained or a list of such.}
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
  \item{verbose}{\code{\link{integer}} verbose level. Can be
    0 (no output), 1 (epoch number and progress bar during each epoch
    with loss), 2 (epoch number and loss), 3 (epoch number).}
  \item{file}{\code{NULL} or a \code{\link{character}} string
    specifying the file in which the trained GNN(s) is (are)
    saved. If \code{file} is provided and the specified file exists,
    it is loaded and returned via \code{\link{loadGNN}()}.}
  \item{\dots}{additional arguments passed to the underlying
    \code{fit()} (which is \code{keras:::fit.keras.engine.training.Model()}).}
}
\value{
  \describe{
    \item{fitGNN()}{the trained \code{x}.}
    \item{fitGNNonce()}{object of class as \code{x} with the GNNs
      trained.}
    \item{is.trained.gnn_GNN()}{\code{\link{logical}}
      indicating whether \code{x} is trained.}
    \item{is.trained.list()}{\code{\link{logical}} of length
      \code{length(x)} indicating, for each component, whether
      it is trained.}
  }
}
%% \details{
%%   In comparison to an untrained GNN, the components \code{model} (keras model),
%%   \code{n.train} (number of training samples), \code{batch.size} (batch size),
%%   \code{n.epoch} (number of epochs for training) and \code{loss}
%%   (the loss function attained per epoch) are altered.
%% }
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
stopifnot(is.trained(GMMN)) # check if trained
## Note: Obviously, in a real-world application, batch.size and n.epoch
##       should be (much) larger (e.g., batch.size = 5000, n.epoch = 300).

## Plot the loss function
plot_loss(GMMN)

## Evaluate (roughly picks up the shape even with our bad choices of
## batch.size and n.epoch)
set.seed(271)
V <- rGNN(GMMN, size = 2000) # sample
plot(V, xlab = expression(V[1]), ylab = expression(V[2]))
}
}
\keyword{optimize}
