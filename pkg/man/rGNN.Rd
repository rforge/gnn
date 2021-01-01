\name{rGNN}
\alias{rGNN}
\alias{rGNN.gnn_GNN}
\title{Sampling from a Generative Neural Network}
\description{
  Sampling method for objects of \code{\link{S3}} class \code{"gnn_GNN"}.
}
\usage{
\method{rGNN}{gnn_GNN}(x, size, prior = NULL, ...)
}
\arguments{
  \item{x}{object of \code{\link{S3}} class \code{"gnn_GNN"}.}
  \item{size}{sample size, a positive \code{\link{integer}}. Ignored
    if \code{prior} is a \code{\link{matrix}}.}
  \item{prior}{one of
    \describe{
      \item{\code{NULL}}{the default, generates independent N(0,1) samples.}
      \item{\code{\link{matrix}}}{passes this
	given sample through the GNN \code{x}. Such a matrix is
        internally and typically obtained via \code{\link{rPrior}()}.}
  }}
  \item{\dots}{additional arguments passed to the underlying
    \code{\link{rPrior}()} if \code{prior = NULL}.}
}
\value{
  \code{(size, dim(x)[1])}-\code{\link{matrix}} of samples.
}
\author{Marius Hofert}
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
## Define dummy model
d <- 2 # bivariate case
GMMN <- GMMN(c(d, 300, d)) # define a GMMN model (with random weights)

## Sampling
n <- 3
(X1 <- rGNN(GMMN, size = n)) # default (independent N(0,1) samples as prior)
(X2 <- rGNN(GMMN, size = n, # passing additional arguments to rPrior()
            qmargins = qexp, method = "sobol", seed = 271))
(X3 <- rGNN(GMMN, prior = matrix(rexp(n * d), ncol = d))) # providing 'prior'
stopifnot(dim(X1) == c(n, d), dim(X2) == c(n, d), dim(X3) == c(n, d))
}
}
\keyword{methods}
