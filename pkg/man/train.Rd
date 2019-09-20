\name{training}
\alias{train}
\alias{train_once}
\title{Functions for Training of Generative Neural Networks}
\description{
  Functions for training generative neural networks.
}
\usage{
train(gnn, data, batch.size, nepoch)
train_once(gnn, data, batch.size, nepoch,
           file, name = rm_ext(basename(file)), package = NULL)
}
\arguments{
  \item{gnn}{GNN object as created by \code{\link{GMMN_model}()} or
    \code{\link{VAE_model}()}.}
  \item{data}{\eqn{(n,d)}-matrix containing the \eqn{n}
    \eqn{d}-dimensional observations of the training data.}
  \item{batch.size}{number of samples used per stochastic gradient step.}
  \item{nepoch}{number of epochs (one epoch equals one pass through
    the complete training dataset while updating the GNN's parameters
    through stochastic gradient steps).}
  \item{file}{\code{\link{character}} string (with or without ending
    \code{.rda}) specifying the file to save the trained GNN to.}
  \item{name}{name under which the trained GNN is saved in \code{file}.}
  \item{package}{name of the package from which to read the trained GNN;
    if \code{\link{NULL}} (the default) the current working directory is used.}
}
\value{
  \describe{
    \item{\code{train()}:}{The trained GNN object.}
    \item{\code{train_once()}:}{If object \code{name} exists in
      \code{file}, \code{train_once()} reads it,
      converts it to a callable GNN object via
      \code{\link{to_callable}()} and returns it.
      Otherwise, \code{train_once()} calls \code{train()}
      to train the GNN, converts it to a savable
      GNN object via \code{\link{to_savable}()}, saves it
      and returns the trained GNN.}
  }
}
\author{Marius Hofert}
\seealso{
  \code{\link{GMMN_model}()}, \code{\link{VAE_model}()},
  \code{\link{to_savable}()}, \code{\link{to_callable}()}.
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
dim <- c(d, 300, d) # dimensions of the input, hidden and output layers
GMMN.mod <- GMMN_model(dim)
GMMN.trained <- train(GMMN.mod, data = U, batch.size = 500, nepoch = 2)
## Note: Obviously, in a real-world application, batch.size and nepoch
##       should be (much) larger (e.g., batch.size = 5000, nepoch = 300).

## Evaluate (roughly picks up the shape even with our bad choices of
## batch.size and nepoch)
set.seed(271)
N.prior <- matrix(rnorm(2000 * d), ncol = d)
V <- predict(GMMN.trained[["model"]], x = N.prior)
plot(V, xlab = expression(V[1]), ylab = expression(V[2]))

## Convert the trained neural network to one that can be saved
## and save it (here: to some temporary file)
GMMN.savable <- to_savable(GMMN.trained)
file <- tempfile("trained_GMMN", fileext = ".rda")
save_rda(GMMN.savable, file = file, names = "GMMN")
}
}
\keyword{optimize}
