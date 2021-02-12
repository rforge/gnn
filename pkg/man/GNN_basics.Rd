\name{GNN_basics}
\alias{print.gnn_GNN}
\alias{str.gnn_GNN}
\alias{summary.gnn_GNN}
\alias{dim.gnn_GNN}
\alias{is.GNN}
\alias{is.GNN.gnn_GNN}
\alias{is.GNN.list}
\title{Basic Functions and Methods}
\description{
  Basic functions and methods for objects of \code{\link{S3}} class \code{"gnn_GNN"}.
}
\usage{
\method{print}{gnn_GNN}(x, \dots)
\method{str}{gnn_GNN}(object, \dots)
\method{summary}{gnn_GNN}(object, \dots)
\method{dim}{gnn_GNN}(x)
\method{is.GNN}{gnn_GNN}(x)
\method{is.GNN}{list}(x)
}
\arguments{
  \item{x}{
    \describe{
      \item{print(), dim()}{object of \code{\link{S3}} class
	\code{"gnn_GNN"}.}
      \item{is.GNN()}{object of \code{\link{S3}} class \code{"gnn_GNN"}
	or a list of such.}
    }
  }
  \item{object}{object of \code{\link{S3}} class \code{"gnn_GNN"}.}
  \item{\dots}{currently not used.}
}
\value{
  \describe{
    \item{print()}{return value of the \code{\link{print}()}
      method for objects of class \code{"\link{list}"}.}
    \item{str()}{nothing, as \code{\link{str}()} returns
      nothing when applied to objects of class \code{"\link{list}"}.}
    \item{summary()}{return value of the \code{\link{summary}()}
      method for objects of class \code{"\link{list}"}.}
    \item{dim()}{slot \code{dim} of \code{x}, so a vector of
      dimensions of input, hidden and output layers.}
    \item{is.GNN()}{\code{\link{logical}} of length equal to the
      length of \code{x} indicating, for each component,
      whether it is an object of class \code{"gnn_GNN"}.}
  }
}
\author{Marius Hofert}
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
d <- 2
dim <- c(d, 300, d) # dimensions of the input, hidden and output layers
GMMN <- FNN(dim) # define the GMMN model
stopifnot(is.GNN(GMMN)) # check for being a GNN
GMMN # print() method
str(GMMN) # str() method
summary(GMMN) # summary() method
stopifnot(dim(GMMN) == c(d, 300, d)) # dim() method
}
}
\keyword{methods}
