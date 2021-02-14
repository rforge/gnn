\name{loss}
\alias{loss}
\title{Loss Function}
\description{
  Implementation of various loss functions to measure
  statistical discrepancy between two datasets.
}
\usage{
loss(x, y, type = c("MMD", "CvM", "MSE", "BCE"), ...)
}
\arguments{
  \item{x}{2d tensor with shape (batch size, dimension of input dataset).}
  \item{y}{2d tensor with shape (batch size, dimension of input dataset).}
  \item{type}{\code{\link{character}} string indicating the type of
    loss used. Currently available are the
    (kernel) maximum mean discrepancy (\code{"MMD"}),
    the Cramer-von Mises statistc (\code{"CvM"}) of R\enc{é}{e}millard
    and Scaillet (2009),
    the mean squared error (\code{"MSE"})
    and the binary cross entropy (\code{"BCE"}).}
  \item{\dots}{additional arguments passed to the underlying loss function,
    e.g. \code{bandwidth}.}
}
\value{
  \code{loss()} returns a 0d tensor containing the loss.
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
