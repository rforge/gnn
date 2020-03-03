\name{human_time}
\alias{human_time}
\title{Time Measurement with Human-Readable Units}
\description{
  \code{\link{system.time}()} with human-readable output.
}
\usage{
human_time(..., digits = 2)
}
\arguments{
  \item{\dots}{arguments passed to the underlying \code{\link{system.time}()}.}
  \item{digits}{for rounding the output; see \code{\link{round}()}.}
}
\value{
  Timings with units indicated.
}
\author{Marius Hofert}
\examples{
human_time(Sys.sleep(1))
}
\keyword{utilities}
