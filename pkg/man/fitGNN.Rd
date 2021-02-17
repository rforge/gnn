\name{fitGNN}
\alias{fitGNN}
\alias{fitGNN.gnn_GNN}
\alias{fitGNNonce}
\alias{fitGNNonce.gnn_GNN}
\alias{is.trained}
\alias{is.trained.gnn_GNN}
\alias{is.trained.list}
\title{Functions and Methods for Training of Generative Neural Networks}
\description{
  Functions and methods for training generative neural networks.
}
\usage{
\method{fitGNN}{gnn_GNN}(x, data, batch.size = nrow(data), n.epoch = 100,
    prior = NULL, max.n.prior = 5000, verbose = 2, ...)
\method{fitGNNonce}{gnn_GNN}(x, data, batch.size = nrow(data), n.epoch = 100,
    prior = NULL, verbose = 2, file = NULL, name = NULL, ...)
\method{is.trained}{gnn_GNN}(x)
\method{is.trained}{list}(x)
}
\arguments{
  \item{x}{
    \describe{
      \item{fitGNN(), fitGNNonce(), is.trained.gnn_GNN()}{object of class
	\code{"gnn_GNN"} to be trained.}
      \item{is.trained.gnn_GNN()}{object of class
	\code{"gnn_GNN"} to be trained or a list of such.}
    }
  }
  \item{data}{\eqn{(n, d)}-matrix containing the \eqn{n}
    \eqn{d}-dimensional observations of the training data.}
  \item{batch.size}{number of samples used per stochastic gradient step.}
  \item{n.epoch}{number of epochs (one epoch equals one pass through
    the complete training dataset while updating the GNN's parameters
    through stochastic gradient steps).}
  \item{prior}{\eqn{(n, d)}-matrix of prior samples; see also
    \code{\link{rPrior}()}. If \code{prior = NULL} a sample of
    independent N(0,1) random variates is generated.}
  \item{max.n.prior}{maximum number of prior samples stored in \code{x}
    after training.}
  \item{verbose}{\code{\link{integer}} verbose level. Choices are:
    \describe{
      \item{0}{silent (no output).}
      \item{1}{progress bar (via \code{\link{txtProgressBar}()}).}
      \item{2}{output after each block of epochs (block size is
	\code{ceiling(n.epoch/10)} if \code{n.epoch <= 100} and
	\code{ceiling(sqrt(n.epoch))} if \code{n.epoch > 100}).}
      \item{3}{output after each expoch.}
    }
  }
  \item{file}{\code{NULL} or a \code{\link{character}} string
    specifying the file in which the trained GNN(s) is (are)
    saved. If \code{file} is provided and the specified file exists,
    it is loaded and returned via \code{\link{loadGNN}()}.}
  \item{name}{\code{\link{character}} string giving the name under
    which the fitted \code{x} is saved (if \code{NULL} the fitted
    \code{x} is saved under the name \code{"x"}).}
  \item{\dots}{additional arguments passed to the underlying
    \code{fit()} (which is \code{keras:::fit.keras.engine.training.Model()}).}
}
\value{
  \describe{
    \item{fitGNN()}{the trained \code{x}.}
    \item{fitGNNonce()}{object of class as \code{x} with the trained GNN.}
    \item{is.trained.gnn_GNN()}{\code{\link{logical}}
      indicating whether \code{x} is trained.}
    \item{is.trained.list()}{\code{\link{logical}} of length
      \code{length(x)} indicating, for each component, whether
      it is trained.}
  }
}
%% \details{
%%   In comparison to an untrained GNN, the components \code{model} (keras model),
%%   \code{n.train} (number of training samples), \code{batch.size} (batch size),
%%   \code{n.epoch} (number of epochs for training) and \code{loss}
%%   (the loss function attained per epoch) are altered.
%% }
\author{Marius Hofert}
\seealso{
  \code{\link{FNN}()}, \code{\link{saveGNN}()},
  \code{\link{loadGNN}()}.
}
\keyword{optimize}
