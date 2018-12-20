\name{loss}
\alias{loss}
\title{Loss Function}
\description{
  Implementation of various loss functions to measure
  statistical discrepancy between two datasets.
}
\usage{
loss(x, y, type = c("MSE", "binary.cross", "MMD"), ...)
}
\arguments{
  \item{x}{2d tensor with shape (batch size, dimension of input dataset).}
  \item{y}{2d tensor with shape (batch size, dimension of input dataset).}
  \item{type}{\code{\link{character}} string indicating the type of
    loss used. Currently available are the mean
    squared error (\code{"MSE"}), binary cross entropy
    (\code{"binary.cross"})
    and (kernel) maximum mean discrepancy (\code{"MMD"}).}
  \item{\dots}{additional arguments passed to the underlying loss function;
    at the moment, this is only affects \code{type = "MMD"} for which
    \code{"bandwidth"} can be provided.}
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
}
\seealso{
  \code{\link{GMMN_model}()} and \code{\link{VAE_model}()} where
  \code{loss()} is used.
}
\keyword{univar}
