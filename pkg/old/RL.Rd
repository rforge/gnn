\name{RL}
\alias{RL}
\title{Reconstruction Loss Function}
\description{
  Implementation of various reconstruction loss functions to measure
  statistical discrepancy between two datasets.
}
\usage{
RL(x, y, type = c("MSE", "binary.cross", "MMD"), ...)
}
\arguments{
  \item{x}{2d tensor with shape (batch size, dimension of input dataset).}
  \item{y}{2d tensor with shape (batch size, dimension of input dataset).}
  \item{type}{\code{\link{character}} string indicating the type of
    reconstruction loss used. Currently available are the mean
    squared error (\code{"mse"}), binary cross entropy
    (\code{"binary.cross"}.)
    and (kernel) maximum mean discrepancy (\code{"mmd"}).}
  \item{\dots}{additional arguments passed to the underlying loss function;
    at the moment, only the bandwith parameter \code{bw} of
    the \code{\link{MMD}()} reconstruction loss function is supported.}
}
\value{
  \code{RL()} returns a 0d tensor containing the reconstruction loss.
}
\author{Marius Hofert and Avinash Prasad}
\references{
  Kingma, D. P. and Welling, M. (2014).
  Stochastic gradient VB and the variational auto-encoder.
  \emph{Second International Conference on Learning Representations (ICLR)}.
  See https://keras.rstudio.com/articles/examples/variational_autoencoder.html
}
\seealso{\code{\link{VAE_model}()}, \code{\link{MMD}()}}
\keyword{univar}
