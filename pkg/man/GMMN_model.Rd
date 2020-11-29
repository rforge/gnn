\name{GMMN_model}
\alias{GMMN_model}
\title{Generative Moment Matching Network}
\description{
  Setup of a generative moment matching network (GMMN) model.
}
\usage{
GMMN_model(dim, activation = c(rep("relu", length(dim) - 2), "sigmoid"),
           batch.norm = FALSE, dropout.rate = 0, nGPU = 0, ...)
}
\arguments{
  \item{dim}{\code{\link{numeric}} vector of length at least two, giving
    the dimensions of the input layer, the hidden layer(s) (if any) and
    the output layer (in this order).}
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
  \code{GMMN_model()} returns a list with components
  \describe{
    \item{\code{model}:}{GMMN model (a \pkg{keras} object inheriting from
      the classes \code{"keras.engine.training.Model"},
      \code{"keras.engine.network.Network"},
      \code{"keras.engine.base_layer.Layer"}
      and \code{"python.builtin.object"}).}
    \item{\code{type}:}{\code{\link{character}} string indicating
      the type of model (\code{"GMMN"}).}
    \item{\code{dim}:}{see above.}
    \item{\code{activation}:}{see above.}
    \item{\code{batch.norm}:}{see above.}
    \item{\code{dropout.rate}:}{see above.}
    \item{\code{dim.train}:}{dimension of the training data (\code{NA}
      unless trained).}
    \item{\code{batch.size}:}{batch size (\code{NA} unless trained).}
    \item{\code{nepoch}:}{number of epochs (\code{NA} unless trained).}
  }
}
\author{Marius Hofert and Avinash Prasad}
\references{
  Li, Y., Swersky, K. and Zemel, R. (2015).
  Generative moment matching networks.
  \emph{Proceedings of Machine Learning Research}, \bold{37}
  (International Conference on Maching Learning), 1718--1727.
  See http://proceedings.mlr.press/v37/li15.pdf (2019-08-24)

  Dziugaite, G. K., Roy, D. M. and Ghahramani, Z. (2015).
  Training generative neural networks via maximum mean discrepancy
  optimization. \emph{AUAI Press}, 258--267.
  See http://www.auai.org/uai2015/proceedings/papers/230.pdf (2019-08-24)
}
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
## Training data
d <- 2 # bivariate case
P <- matrix(0.9, nrow = d, ncol = d); diag(P) <- 1 # correlation matrix
ntrn <- 60000 # training data sample size
set.seed(271)
library(mvtnorm)
X <- rmvnorm(ntrn, sigma = P) # N(0,P) samples
X. <- abs(X) # |X|

## Plot a subsample
m <- 2000 # subsample size for plots
opar <- par(pty = "s")
plot(X.[1:m,], xlab = expression(X[1]), ylab = expression(X[2])) # plot |X|
U <- apply(X., 2, rank) / (ntrn + 1) # pseudo-observations of |X|
plot(U[1:m,], xlab = expression(U[1]), ylab = expression(U[2])) # visual check

## Define the model and 'train' it
dim <- c(d, 300, d) # dimensions of the input, hidden and output layers
GMMN.mod <- GMMN_model(dim) # define the GMMN model
nbat <- 500 # batch size = number of samples per gradient step (120x steps/epoch)
nepo <- 10 # number of epochs = number of times training data is shuffled
GMMN <- train(GMMN.mod, data = U, batch.size = nbat, nepoch = nepo)
## Note:
## - Obviously, in a real-world application, batch.size and nepoch
##   should be (much) larger (e.g., batch.size = 5000, nepoch = 300).
## - The above training is not reproducible (due to keras).

## Evaluate GMMN based on prior sample (already roughly picks up the shape)
set.seed(271)
N.prior <- matrix(rnorm(m * d), ncol = d) # sample from the prior distribution
V <- predict(GMMN[["model"]], x = N.prior) # feedforward through GMMN

## Joint plot of training subsample with GMMN PRNs
layout(t(1:2))
plot(U[1:m,], xlab = expression(U[1]), ylab = expression(U[2]), cex = 0.2)
plot(V,       xlab = expression(V[1]), ylab = expression(V[2]), cex = 0.2)

## Joint plot of training subsample with GMMN QRNs
library(qrng) # for sobol()
V. <- predict(GMMN[["model"]], x = qnorm(sobol(m, d = d, randomize = "Owen",
                                               seed = 271)))
plot(U[1:m,], xlab = expression(U[1]), ylab = expression(U[2]), cex = 0.2)
plot(V.,      xlab = expression(V[1]), ylab = expression(V[2]), cex = 0.2)
layout(1)
par(opar)
}
}
\keyword{datagen}
