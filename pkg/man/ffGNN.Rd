\name{ffGNN}
\alias{ffGNN}
\alias{ffGNN.gnn_GNN}
\title{Feedforward for Generative Neural Networks}
\description{
  Feedforward method for objects of \code{\link{S3}} class \code{"gnn_GNN"}.
}
\usage{
\method{ffGNN}{gnn_GNN}(x, data)
}
\arguments{
  \item{x}{object of \code{\link{S3}} class \code{"gnn_GNN"}.}
  \item{data}{\code{\link{matrix}} to be fed forward through \code{x}.}
}
\value{
  The output \code{\link{matrix}} of \code{x} when fed with \code{data}.
}
\author{Marius Hofert}
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
## Define dummy model
d <- 2 # bivariate case
GMMN <- GMMN(c(d, 300, d)) # define a GMMN model (with random weights)

## Feedforward
n <- 3
set.seed(271)
X <- ffGNN(GMMN, data = matrix(runif(n * d), ncol = d))
stopifnot(dim(X) == c(n, d))
}
}
\keyword{methods}
