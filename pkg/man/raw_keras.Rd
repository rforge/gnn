\name{raw_keras}
\alias{as.keras}
\alias{as.raw.gnn_GNN}
\alias{as.keras.gnn_GNN}
\title{Convert GNN model Slots to raw or keras Objects}
\description{
  Keras objects cannot be saved like other R objects.
  The methods \code{as.raw()} and \code{as.keras()} can
  be used to convert the \code{model} slots of objects
  of \code{\link{S3}} class \code{"gnn_GNN"} to
  \code{"\link{raw}"} objects (which can be saved)
  or \code{"keras.engine.training.Model"} objects (which
  can be trained).
}
\usage{
\method{as.raw}{gnn_GNN}(x)
\method{as.keras}{gnn_GNN}(x)
}
\arguments{
  \item{x}{object of \code{\link{S3}} class \code{"gnn_GNN"}.}
}
\value{
  object of \code{\link{S3}} class \code{"gnn_GNN"} with
  slot \code{method} converted by the respective method if necessary.
}
%% \details{
%% For VAEs, \code{to_savable()} is (indirectly) based on
%% \code{\link{save_model_weights_hdf5}()} and \code{to_callable()}
%% on \code{\link{load_model_weights_hdf5}()}; one cannot work
%% with \code{\link{serialize_model}()} or
%% \code{\link{unserialize_model}()} in this case because of the
%% involved \code{\link{layer_lambda}()}.
%% }
\author{Marius Hofert}
\keyword{methods}
