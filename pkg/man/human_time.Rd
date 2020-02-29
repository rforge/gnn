\name{human_time}
\alias{human_time}
\title{Time Measurement with Human-Readable Units}
\description{
  \code{\link{system.time}()} with human-readable output.
}
\usage{
human_time(expr, ...)
}
\arguments{
  \item{expr}{see \code{\link{system.time}()}.}
  \item{\dots}{additional arguments passed to the underlying
    \code{\link{system.time}()}.}
}
\value{
  Timings with units indicated.
}
\author{Marius Hofert}
\examples{
human_time(Sys.sleep(1))
}
\keyword{utilities}
