\name{time}
\alias{as.human}
\alias{time.gnn_GNN}
\title{Training Time in Human-Readable Format}
\description{
  Functions and methods for extracting training time in
  human-readable format.
}
\usage{
as.human(x, fmt = "\%.2f")
\method{time}{gnn_GNN}(x, human = FALSE, ...)
}
\arguments{
  \item{x}{
    \describe{
      \item{as.human()}{object of class \code{"proc_time"}, for example
	produced by a call to \code{\link{system.time}()}.}
      \item{time()}{object of class \code{"gnn_GNN"}.}
    }
  }
  \item{fmt}{format string as required by \code{\link{sprintf}()}.}
  \item{human}{logical indicating whether the result is to be returned
    in human-readable format.}
  \item{\dots}{additional arguments passed to the underlying
    \code{as.human()}.}
}
\value{
  \describe{
    \item{as.human()}{named \code{\link{character}(3)} providing
      user, system and elapsed time in human-readable format.}
    \item{time()}{named \code{\link{character}(3)} (invisibly) providing
      user, system and elapsed time; prints these times by side-effect.}
  }
}
\author{Marius Hofert}
\keyword{utilities}
