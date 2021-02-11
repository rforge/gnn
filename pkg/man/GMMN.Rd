\name{GMMN}
\alias{GMMN}
\title{Generative Moment Matching Network}
\description{
  Constructor for a generative moment matching network (GMMN) model,
  an object of \code{\link{S3}} class \code{"gnn_GMMN"}.
}
\usage{
GMMN(dim = c(2, 2), activation = c(rep("relu", length(dim) - 2), "sigmoid"),
     batch.norm = FALSE, dropout.rate = 0, n.GPU = 0, ...)
}
\arguments{
  \item{dim}{\code{\link{integer}} vector of length at least two, giving
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
  \item{n.GPU}{non-negative \code{\link{integer}} specifying the number of GPUs
    available if the GPU version of TensorFlow is installed.
    If positive, a (special) multiple GPU model for data
    parallelism is instantiated. Note that for multi-layer perceptrons
    on a few GPUs, this model does not yet yield any scale-up computational
    factor (in fact, currently very slightly negative scale-ups are likely due
    to overhead costs).}
  \item{\dots}{additional arguments passed to \code{\link{loss}()}.}
}
\value{
  \code{GMMN()} returns an object of \code{\link{S3}} class \code{"gnn_GMMN"}
  with components
  \describe{
    \item{\code{model}}{GMMN model (a \pkg{keras} object inheriting from
      the R6 classes \code{"keras.engine.training.Model"},
      \code{"keras.engine.network.Network"},
      \code{"keras.engine.base_layer.Layer"}
      and \code{"python.builtin.object"}, or a \code{\link{raw}}
      object).}
    \item{\code{type}}{\code{\link{character}} string indicating
      the type of model (\code{"GMMN"}).}
    \item{\code{dim}}{see above.}
    \item{\code{activation}}{see above.}
    \item{\code{batch.norm}}{see above.}
    \item{\code{dropout.rate}}{see above.}
    \item{\code{n.param}}{number of trainable, non-trainable and total
      number of parameters.}
    \item{\code{n.train}}{number of training samples (\code{NA_integer_}
      unless trained).}
    \item{\code{batch.size}}{batch size (\code{NA_integer_} unless trained).}
    \item{\code{n.epoch}}{number of epochs (\code{NA_integer_} unless trained).}
  }
}
\details{
  The \code{\link{S3}} class \code{"gnn_GMMN"} is a subclass of the
  \code{\link{S3}} class \code{"gnn_GNN"} which in turn is a subclass of
  \code{"gnn_Model"}.
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

  Hofert, M., Prasad, A. and Zhu, M. (2020).
  Quasi-random sampling for multivariate distributions via generative
  neural networks. \emph{Journal of Computational and Graphical
  Statistics}, \doi{10.1080/10618600.2020.1868302}.

  Hofert, M., Prasad, A. and Zhu, M. (2020).
  Multivariate time-series modeling with generative neural networks.
  See \url{https://arxiv.org/abs/2002.10645}.

  Hofert, M. Prasad, A. and Zhu, M. (2020).
  Applications of multivariate quasi-random sampling with neural
  networks. See \url{https://arxiv.org/abs/2012.08036}.
}
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
## Training data
d <- 2 # bivariate case
P <- matrix(0.9, nrow = d, ncol = d); diag(P) <- 1 # correlation matrix
ntrn <- 60000 # training data sample size
set.seed(271)
library(nvmix)
X <- rNorm(ntrn, scale = P) # N(0,P) samples
X. <- abs(X) # |X|

## Plot a subsample
m <- 2000 # subsample size for plots
opar <- par(pty = "s")
plot(X.[1:m,], xlab = expression(X[1]), ylab = expression(X[2])) # plot |X|
U <- apply(X., 2, rank) / (ntrn + 1) # pseudo-observations of |X|
plot(U[1:m,], xlab = expression(U[1]), ylab = expression(U[2])) # visual check

## Define the model
GMMN <- GMMN(c(d, 300, d)) # define a GMMN model

## Define the batch size and maximal number of epochs
batch <- 500 # batch size = number of samples per gradient step (120x steps/epoch)
n.epo <- 10 # number of epochs = number of times training data is shuffled

## Train or fit
library(keras) # for callback_early_stopping()
## We monitor the loss function and stop earlier (before n.epo-many epochs)
## if the loss function over the last patience-many epochs has changed by
## less than min_delta in absolute value. Then we keep the weights that
## led to the smallest loss seen throughout training.
GMMN <- fitGNN(GMMN, data = U, batch.size = batch, n.epoch = n.epo,
               callbacks = callback_early_stopping(monitor = "loss",
                                                   min_delta = 1e-3, patience = 3,
                                                   restore_best_weights = TRUE))
stopifnot(is.trained(GMMN)) # check if trained
## Note:
## - Obviously, in a real-world application, batch.size and n.epoch
##   should be (much) larger (e.g., batch.size = 5000, n.epoch = 300).
## - The above training is not reproducible (due to keras).

## Evaluate GMMN based on prior sample (already roughly picks up the shape)
set.seed(271)
V <- rGNN(GMMN, size = m) # sample

## Joint plot of training subsample with GMMN PRNs
layout(t(1:2))
plot(U[1:m,], xlab = expression(U[1]), ylab = expression(U[2]), cex = 0.2)
plot(V,       xlab = expression(V[1]), ylab = expression(V[2]), cex = 0.2)

## Joint plot of training subsample with GMMN QRNs
library(qrng) # for sobol()
V. <- rGNN(GMMN, size = m, method = "sobol", randomize = "Owen")
plot(U[1:m,], xlab = expression(U[1]), ylab = expression(U[2]), cex = 0.2)
plot(V.,      xlab = expression(V[1]), ylab = expression(V[2]), cex = 0.2)
layout(1)
par(opar)

## Plot the loss function values after each epoch
plot_loss(GMMN)
}
}
\keyword{models}
