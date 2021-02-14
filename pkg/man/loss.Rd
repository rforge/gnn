\name{loss}
\alias{loss}
\alias{MMD}
\alias{CvM}
\title{Loss Function}
\description{
  Implementation of various loss functions to measure
  statistical discrepancy between two datasets.
}
\usage{
loss(x, y, type = c("MMD", "CvM", "MSE", "BCE"), ...)

MMD(x, y, ...)
CvM(x, y)
}
\arguments{
  \item{x}{2d-tensor or \eqn{(n, d)}-matrix (during training, \eqn{n} is
    the batch size and \eqn{d} is the dimension of the input dataset).}
  \item{y}{2d-tensor or \eqn{(m, d)}-matrix (during training, \eqn{m} is
    the batch size (and typically equal to \eqn{n}) and \eqn{d} is the
    dimension of the input dataset).}
  \item{type}{\code{\link{character}} string indicating the type of
    loss used. Currently available are the
    (kernel) maximum mean discrepancy (\code{"MMD"}, calling \code{MMD()}),
    the Cramer-von Mises statistc (\code{"CvM"}, calling \code{CvM()})
    of R\enc{é}{e}millard and Scaillet (2009),
    the mean squared error (\code{"MSE"})
    and the binary cross entropy (\code{"BCE"}).}
  \item{\dots}{additional arguments passed to the underlying functions,
    most notably \code{bandwidth} (a number or numeric vector of
    bandwidths for the radial basis function kernels) in case
    \code{type = "MMD"}.}
}
\value{
  \code{loss()} returns a 0d tensor containing the loss.

  \code{MMD()} and \code{CvM()} return a 0d tensor (if \code{x}
  and \code{y} are tensors) or \code{\link{numeric}(1)} (if \code{x} or
  \code{y} are \R matrices).
}
\author{Marius Hofert and Avinash Prasad}
\references{
  Kingma, D. P. and Welling, M. (2014).
  Stochastic gradient VB and the variational auto-encoder.
  \emph{Second International Conference on Learning Representations (ICLR)}.
  See https://keras.rstudio.com/articles/examples/variational_autoencoder.html

  R\enc{é}{e}millard, B. and Scaillet, O. (2009).
  Testing for equality between two copulas.
  \emph{Journal of Multivariate Analysis}
  \bold{100}, 377--386.
}
\seealso{
  \code{\link{FNN}()} where \code{loss()} is used.
}
\keyword{univar}
