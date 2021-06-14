\name{gof2sample}
\alias{gof2sample}
\alias{print.htest2}
\title{Two-Sample Goodness-of-Fit Test}
\description{
  Two-sample goodness-of-fit test for equality in distribution based on
  a bootstrap.
}
\usage{
gof2sample(x, y, B = 1000, method = c("MMD", "CvM"), progress = TRUE, ...)
\method{print}{htest2}(x, digits = getOption("digits"), prefix = "\t", ...)
}
\arguments{
  \item{x,y}{datasets (\code{\link{matrix}}) to compare. For
    \code{print.htest2()}, \code{x} is an object of class \code{"htest2"}.}
  \item{B}{number ob bootstrap replications.}
  \item{method}{\code{\link{character}} string indicating the type of
    test statistic to be used (currently the Maximum Mean Discrepancy
    (MMD) or the Cram\'er-von Mises (CvM) statistics).}
  \item{progress}{\code{\link{logical}} indicating whether progress
    output is to be written.}
  \item{digits}{number of digits to consider for formatting the output.}
  \item{prefix}{\code{\link{character}} string to be printed before the
    method.}
  \item{\dots}{
    \describe{
      \item{\code{gof2sample()}}{additional arguments to be passed to
	the underlying \code{method}.}
      \item{\code{print.htest2()}}{unused (for compatibility with
	\code{\link{print}()} method.}
    }
  }
}
\value{
  \describe{
    \item{\code{gof2sample()}}{an object of class \code{"htest2"}.}
    \item{\code{print.htest2()}}{\code{x} invisibly.}
  }
}
\author{Marius Hofert}
\seealso{
  \code{getAnywhere("print.htest")}
}
\examples{
n <- 18
m <- 22
d <- 2
set.seed(271)
x <- matrix(rnorm(n * d), ncol = d)
y <- matrix(rt(m * d, df = 3), ncol = d)
gof2sample(x, y, B = 10)
}
\keyword{htest}
\keyword{goodness-of-fit}
