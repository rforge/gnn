\name{rPrior}
\alias{rPrior}
\title{Sampling from a Prior Distribution}
\description{
  Sampling from a prior distribution.
}
\usage{
rPrior(n, copula, qmargins = qnorm, method = c("pseudo", "sobol"), ...)
}
\arguments{
  \item{n}{sample size, a positive \code{\link{integer}}.}
  \item{copula}{object of \code{\link{S4}} class \code{"Copula"} for
    which the method \code{\link{rCopula}()} is available; see the \R
    package \pkg{copula}.}
  \item{qmargins}{marginal quantile \code{\link{function}} or a \code{\link{list}}
    of length \code{dim(x)[1]} of such.}
  \item{method}{\code{\link{character}} string indicating the sampling
    method. If \code{"sobol"}, then randomization \code{"digital.shift"}
    is used (pass \code{seed} via \dots for reproducibilty;
    see the \R package \pkg{qrng}).}
  \item{\dots}{additional arguments passed to \code{method}.}
}
\value{
  \code{(n, dim(copula))}-\code{\link{matrix}} of samples.
}
\author{Marius Hofert}
\examples{
n <- 5
d <- 3
library(copula)
cop <- claytonCopula(2, dim = d)
X1 <- rPrior(n, copula = cop) # Clayton copula and N(0,1) margins
X2 <- rPrior(n, copula = cop, qmargins = qexp) # Exp(1) margins
X3 <- rPrior(n, copula = cop, qmargins = qexp, method = "sobol", seed = 271)
stopifnot(dim(X1) == c(n, d), dim(X2) == c(n, d), dim(X3) == c(n, d))
}
\keyword{methods}
