\name{GNN_basics}
\alias{print.gnn_GNN}
\alias{str.gnn_GNN}
\alias{summary.gnn_GNN}
\alias{dim.gnn_GNN}
\title{Basic Methods}
\description{
  Basic methods for objects of \code{\link{S3}} class \code{"gnn_GNN"}.
}
\usage{
\method{print}{gnn_GNN}(x, \dots)
\method{str}{gnn_GNN}(object, \dots)
\method{summary}{gnn_GNN}(object, \dots)
\method{dim}{gnn_GNN}(x)
}
\arguments{
  \item{x, object}{object of \code{\link{S3}} class \code{"gnn_GNN"}.}
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
    \item{\code{dim()}}{slot \code{dim} of \code{x}, so a vector of
      dimensions of input, hidden and output layers.}
  }
}
\author{Marius Hofert}
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
d <- 2
dim <- c(d, 300, d) # dimensions of the input, hidden and output layers
GMMN.mod <- GMMN(dim) # define the GMMN model
GMMN.mod # print() method
str(GMMN.mod) # str() method
summary(GMMN.mod) # summary() method
stopifnot(dim(GMMN.mod) == c(d, 300, d)) # dim() method
}
}
\keyword{methods}
