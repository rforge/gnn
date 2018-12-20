\name{VAE_model}
\alias{VAE_model}
\title{Variational Autoencoder}
\description{
  Setup of a variational autoencoder (VAE) model.
}
\usage{
VAE_model(dim, activation = c(rep("relu", length(dim) - 2), "sigmoid"),
          batch.normalization = FALSE, dropout.rate = 0,
          sd = 1, loss.type = c("MSE", "binary.cross", "MMD"), nGPUs = 0, ...)
}
\arguments{
  \item{dim}{\code{\link{numeric}} vector of length at least two, giving
    the dimensions of the input layer (equal to the dimension of the
    output layer), the hidden layer(s) (if any) and the latent layer (in
    this order).}
  \item{activation}{\code{\link{character}} vector of length
    \code{length(dim) - 1} specifying the activation functions
    for all hidden layers and the output layer (in this order);
    note that the input layer does not have an activation function.}
  \item{batch.normalization}{\code{\link{logical}} indicating whether batch
    normalization layers are to be added after each hidden layer.}
  \item{dropout.rate}{\code{\link{numeric}} value in [0,1] specifying
    the fraction of input to be dropped; see the rate parameter of
    \code{\link{layer_dropout}()}. Note that only if positive, dropout
    layers are added after each hidden layer.}
  \item{sd}{positive \code{\link{numeric}} value giving the standard
    deviation of the normal distribution used as prior.}
  \item{loss.type}{\code{\link{character}} string indicating the type of
    reconstruction loss. Currently available are the mean squared error
    (\code{"MSE"}), binary cross entropy (\code{"binary.cross"})
    and (kernel) maximum mean discrepancy (\code{"MMD"}).}
  \item{nGPUs}{non-negative \code{\link{integer}} specifying the number of GPUs
    available if the GPU version of TensorFlow is installed.
    If positive, a (special) multiple GPU model for data
    parallelism is instantiated. Note that for multi-layer perceptrons
    on a few GPUs, this model does not yet yield any scale-up computational
    factor (in fact, currently very slightly negative scale-ups are likely due
    to overhead costs).}
  \item{\dots}{additional arguments passed to \code{\link{loss}()}.}
}
\value{
  \code{VAE_model()} returns a \code{\link{list}} with components
  \code{"model"}, \code{"encoder"} and \code{"generator"} containing the VAE model,
  its encoder and generator.
}
\references{
  Kingma, D. P. and Welling, M. (2014).
  Stochastic gradient VB and the variational auto-encoder.
  \emph{Second International Conference on Learning Representations (ICLR)}.
  See https://keras.rstudio.com/articles/examples/variational_autoencoder.html
}
\author{Marius Hofert and Avinash Prasad}
\seealso{
  \code{\link{GMMN_model}()}
}
\keyword{datagen}