\name{GMMN_basics}
\alias{print.gnn_GMMN}
\alias{str.gnn_GMMN}
\alias{summary.gnn_GMMN}
\title{Basic Methods}
\description{
  Basic methods for objects of \code{\link{S3}} class \code{"gnn_GMMN"}.
}
\usage{
\method{print}{gnn_GMMN}(x, \dots)
\method{str}{gnn_GMMN}(object, \dots)
\method{summary}{gnn_GMMN}(object, \dots)
}
\arguments{
  \item{x, object}{object of \code{\link{S3}} class \code{"gnn_GMMN"}.}
  \item{\dots}{currently not used.}
}
\value{
  \describe{
    \item{\code{print()}}{return value of the \code{\link{print}()}
       method for objects of class \code{"\link{list}"}.}
     \item{\code{str()}}{nothing, as \code{\link{str}()} returns
       nothing when applied to objects of class \code{"\link{list}"}.}
    \item{\code{summary()}}{return value of the \code{\link{summary}()}
       method for objects of class \code{"\link{list}"}.}
  }
}
\author{Marius Hofert and Avinash Prasad}
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
d <- 2
dim <- c(d, 300, d) # dimensions of the input, hidden and output layers
GMMN.mod <- GMMN(dim) # define the GMMN model
GMMN.mod # print() method
str(GMMN.mod) # str() method
summary(GMMN.mod) # summary() method
}
}
\keyword{datagen}
