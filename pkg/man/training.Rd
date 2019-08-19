\name{training}
\alias{train}
\alias{train_once}
\title{Functions for Training of Generative Neural Networks}
\description{
  Functions for training generative neural networks.
}
\usage{
train(dim, data, batch.size, nepoch, method = c("GMMN", "VAE"), ...)
train_once(dim, data, batch.size, nepoch, method = c("GMMN", "VAE"),
           file, name = rm_ext(basename(file)),
           package = NULL, ...)
}
\arguments{
  \item{dim}{\code{\link{numeric}} vector of length at least two, giving
    the dimensions of the input layer, the hidden layer(s) (if any) and
    the output layer (in this order).}
  \item{data}{\eqn{(n,d)}-matrix containing the \eqn{n}
    \eqn{d}-dimensional observations of the training data.}
  \item{batch.size}{number of samples per stochastic gradient step.}
  \item{nepoch}{number of epochs (one epoch equals one pass through
    the complete training dataset while updating the GNN's parameters).}
  \item{method}{\code{\link{character}} string specifying the type of
    generative neural network.}
  \item{file}{\code{\link{character}} string (with or without ending
    \code{.rda}) specifying the file to save the trained GNN to.}
  \item{name}{name under which the trained GNN is saved in \code{file}.}
  \item{package}{package name from which to load the object or
    \code{\link{NULL}} (the default) in which case the current working
    directory is searched.}
  \item{\dots}{for
    \describe{
      \item{\code{train()}}{additional arguments passed to the
	underlying \code{\link{GMMN_model}()} (if \code{model = "GMMN"})
	or \code{\link{VAE_model}()} (if \code{model = "VAE"}).}
      \item{\code{train_once()}}{additional arguments passed to the
	underlying \code{train()}.}
    }
  }
}
\value{
  The trained GNN (an object inheriting from
  the classes \code{"keras.engine.training.Model"},
  \code{"keras.engine.network.Network"},
  \code{"keras.engine.base_layer.Layer"}
  and \code{"python.builtin.object"}).
}
\details{
  If object \code{name} exists in \code{file}, then \code{train_once()}
  simply returns it; otherwise \code{train_once()} calls \code{train()}
  to train the GNN, then saves it via \code{\link{save_rda}()} and
  returns the trained GNN.
}
\author{Marius Hofert}
\seealso{
  \code{\link{GMMN_model}()}, \code{\link{VAE_model}()}
}
\keyword{optimize}
