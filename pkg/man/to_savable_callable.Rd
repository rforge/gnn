\name{to_savable_callable}
\alias{to_savable}
\alias{to_callable}
\title{Convert GNN objects to Savable or Callable Ones}
\description{
  Keras objects cannot be saved like other R objects.
  The auxiliary functions \code{to_savable()} and
  \code{to_callable()} address this issue.
}
\usage{
to_savable(gnn)
to_callable(gnn)
}
\arguments{
  \item{gnn}{GNN object.}
}
\value{
  \describe{
    \item{\code{to_savable()}:}{The GNN object with \pkg{keras}
      components replaced by savable ones.}
    \item{\code{to_callable()}:}{The GNN object with
      certain components replaced by \pkg{keras} objects.}
  }
}
\details{
  For GMMNs, \code{to_savable()} calls \code{\link{serialize_model}()}
  and \code{to_callable()} calls \code{\link{unserialize_model}()}.

  For VAEs, \code{to_savable()} is (indirectly) based on
  \code{\link{save_model_weights_hdf5}()} and \code{to_callable()}
  on \code{\link{load_model_weights_hdf5}()}; one cannot work
  with \code{\link{serialize_model}()} or
  \code{\link{unserialize_model}()} in this case because of the
  involved \code{\link{layer_lambda}()}.

  See the source code for more details.
}
\author{Marius Hofert}
\keyword{utilities}
