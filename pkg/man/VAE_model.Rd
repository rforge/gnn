\name{VAE_model}
\alias{VAE_model}
\title{Variational Autoencoder}
\description{
  Setup of a variational autoencoder (VAE) model.
}
\usage{
VAE_model(dim, activation = c(rep("relu", length(dim) - 2), "sigmoid"),
          batch.norm = FALSE, dropout.rate = 0,
          sd = 1, loss.type = c("MSE", "binary.cross", "MMD"), nGPU = 0, ...)
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
  \item{batch.norm}{\code{\link{logical}} indicating whether batch
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
  \item{nGPU}{non-negative \code{\link{integer}} specifying the number of GPUs
    available if the GPU version of TensorFlow is installed.
    If positive, a (special) multiple GPU model for data
    parallelism is instantiated. Note that for multi-layer perceptrons
    on a few GPUs, this model does not yet yield any scale-up computational
    factor (in fact, currently very slightly negative scale-ups are likely due
    to overhead costs).}
  \item{\dots}{additional arguments passed to \code{\link{loss}()}.}
}
\value{
  \code{VAE_model()} returns a list with components
  \describe{
    \item{\code{model}:}{VAE model (a \pkg{keras} object inheriting from
      the classes \code{"keras.engine.training.Model"},
      \code{"keras.engine.network.Network"},
      \code{"keras.engine.base_layer.Layer"}
      and \code{"python.builtin.object"}).}
    \item{\code{encoder}:}{the encoder (a \pkg{keras} object as
      \code{model}).}
    \item{\code{generator}:}{the generator (a \pkg{keras} object as
      \code{model}).}
    \item{\code{type}:}{\code{\link{character}} string indicating
      the type of model (\code{"VAE"}).}
    \item{\code{dim}:}{see above.}
    \item{\code{activation}:}{see above.}
    \item{\code{batch.norm}:}{see above.}
    \item{\code{dropout.rate}:}{see above.}
    \item{\code{sd}:}{see above.}
    \item{\code{loss.type}:}{see above.}
    \item{\code{dim.train}:}{dimension of the training data (\code{NA}
      unless trained).}
    \item{\code{batch.size}:}{batch size (\code{NA} unless trained).}
    \item{\code{nepoch}:}{number of epochs (\code{NA} unless trained).}
  }
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
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
## Example model with a 5d input, 300d hidden and 4d output layer
str(VAE_model(c(5, 300, 4)))
}
}
\keyword{datagen}