\name{training}
\alias{train}
\alias{train_once}
\title{Functions for Training of Generative Neural Networks}
\description{
  Functions for training generative neural networks.
}
\usage{
train(gnn, data, batch.size, nepoch)
train_once(gnn, data, batch.size, nepoch,
           file, name = rm_ext(basename(file)))
}
\arguments{
  \item{gnn}{GNN object as created by \code{\link{GMMN_model}()} or
    \code{\link{VAE_model}()}.}
  \item{data}{\eqn{(n,d)}-matrix containing the \eqn{n}
    \eqn{d}-dimensional observations of the training data.}
  \item{batch.size}{number of samples used per stochastic gradient step.}
  \item{nepoch}{number of epochs (one epoch equals one pass through
    the complete training dataset while updating the GNN's parameters
    through stochastic gradient steps).}
  \item{file}{\code{\link{character}} string (with or without ending
    \code{.rda}) specifying the file to save the trained GNN to.}
  \item{name}{name under which the trained GNN is saved in \code{file}.}
}
\value{
  \describe{
    \item{\code{train()}:}{The trained GNN object.}
    \item{\code{train_once()}:}{If object \code{name} exists in
      \code{file}, \code{train_once()} reads it,
      converts it to a callable GNN object via
      \code{\link{to_callable}()} and returns it.
      Otherwise, \code{train_once()} calls \code{train()}
      to train the GNN, converts it to a savable
      GNN object via \code{\link{to_savable}()}, saves it
      and returns the trained GNN.}
  }
}
\author{Marius Hofert}
\seealso{
  \code{\link{GMMN_model}()}, \code{\link{VAE_model}()},
  \code{\link{to_savable}()}, \code{\link{to_callable}()}.
}
\keyword{optimize}
