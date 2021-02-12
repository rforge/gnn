\name{time}
\alias{as.human}
\alias{human.time}
\alias{time.gnn_GNN}
\alias{print.gnn_proc_time}
\title{Human-Readable Time Measurement}
\description{
  Functions and methods for extracting and printing timings in
  human-readable format.
}
\usage{
as.human(x, fmt = "\%.2f")
human.time(expr, print = TRUE, ...)
\method{time}{gnn_GNN}(x, human = FALSE, ...)
\method{print}{gnn_proc_time}(x, ...)
}
\arguments{
  \item{x}{
    \describe{
      \item{as.human()}{object of class \code{"proc_time"} as
	returned by \code{\link{system.time}()}.}
      \item{time.gnn_GNN()}{object of class \code{"gnn_GNN"}.}
      \item{print.gnn_proc_time()}{object of class
	\code{"gnn_proc_time"} as returned by \code{time()}.}
    }
  }
  \item{fmt}{format string as required by \code{\link{sprintf}()}.}
  \item{expr}{see \code{\link{system.time}()}.}
  \item{print}{\code{\link{logical}} indicating whether to print the
    result; either way, it is returned (invisibly if \code{print = TRUE}).}
  \item{human}{\code{\link{logical}} indicating whether the result is to
    be returned in human-readable format.}
  \item{\dots}{for \code{human.time()} and \code{time.gnn_GNN()} additional
    arguments passed to the underlying \code{as.human()}; unused for
    \code{print.gnn_proc_time()}.}
}
\value{
  \describe{
    \item{as.human(), human.time()}{named \code{\link{character}(3)} providing
      user, system and elapsed time in human-readable format.}
    \item{time.gnn_GNN()}{object of class \code{"gnn_proc_time"}.}
    \item{print.gnn_proc_time()}{\code{x} (invisibly).}
  }
}
\author{Marius Hofert}
\examples{
human.time(Sys.sleep(0.1)) # print human-readable time
(proc.obj <- human.time(Sys.sleep(0.1), print = FALSE)) # save the timing (character(3))
time(FNN()) # default print method for objects of class "gnn_proc_time"
time(FNN(), human = TRUE) # human-readable print method for such objects
}
\keyword{utilities}
