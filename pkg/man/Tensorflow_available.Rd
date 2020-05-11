\name{TensorFlow_available}
\alias{TensorFlow_available}
\title{A Simple Check whether TensorFlow is Available}
\description{
  A simple (and restrictive) check whether TensorFlow is available.
}
\usage{
TensorFlow_available()
}
\value{
  \code{\link{logical}} indicating whether TensorFlow was found.
}
\details{
  Essentially calls \code{"pip list | grep tensorflow"} via
  \code{\link{system}()}. Only available on non-Windows operating
  systems; returns \code{\link{FALSE}} on Windows.
}
\author{
  Marius Hofert
}
\examples{
TensorFlow_available()
}
\keyword{programming}
