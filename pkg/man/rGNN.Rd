\name{rGNN}
\alias{rGNN}
\alias{rGNN.gnn_GNN}
\title{Sampling from a Generative Neural Network}
\description{
  Sampling method for objects of \code{\link{S3}} class \code{"gnn_GNN"}.
}
\usage{
\method{rGNN}{gnn_GNN}(x, size, prior = NULL, copula = indepCopula(dim(x)[1]),
    qmargin = qnorm, method = c("pseudo", "sobol"), ...)
}
\arguments{
  \item{x}{object of \code{\link{S3}} class \code{"gnn_GNN"}.}
  \item{size}{sample size, a positive \code{\link{integer}}. Ignored
    if \code{prior} is a \code{\link{vector}} or \code{\link{matrix}}.}
  \item{prior}{one of
    \describe{
      \item{\code{NULL}}{the default, generates a sample from the
	copula \code{copula} and margins specified by \code{qmargin}.}
      \item{\code{\link{vector}} or \code{\link{matrix}}}{passes this
	given sample through the GNN \code{x}.}
      \item{\code{\link{function}}}{applies this given function
	to \code{size} to generate the sample to be passed
	through the GNN \code{x}.}
    }}
  \item{copula}{object of \code{\link{S4}} class \code{"Copula"} for
    which the method \code{\link{rCopula}()} is available; see the \R
    package \pkg{copula}.}
  \item{qmargin}{marginal quantile function or a \code{\link{list}}
    of length \code{dim(x)[1]} of such.}
  \item{method}{\code{\link{character}} string indicating the sampling
    method. If \code{"sobol"}, then randomization \code{"digital.shift"}
    is used with seed 271; see the \R package \pkg{qrng}.}
  \item{\dots}{additional arguments passed to \code{method}.}
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
(X1 <- rGNN(GMMN, size = n))
(X2 <- rGNN(GMMN, size = n, method = "sobol"))
(X3 <- rGNN(GMMN, prior = matrix(runif(n * d), ncol = d)))
(X4 <- rGNN(GMMN, size = n,
            prior = function(n.) matrix(runif(n. * d), ncol = d)))
stopifnot(dim(X1) == c(n, d), dim(X2) == c(n, d),
          dim(X3) == c(n, d), dim(X4) == c(n, d))
}
}
\keyword{methods}
