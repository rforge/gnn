\name{save_load_rda}
\alias{saveGNN}
\alias{loadGNN}
\title{Save and Load .rda Files with Conversion to Raw and Keras Models}
\description{
  Save and load \code{.rda} files with conversion to objects of class
  \code{raw} (for \code{saveGNN()}) or \code{"keras.engine.training.Model"}
  (for \code{loadGNN()}).
}
\usage{
saveGNN(..., file, name = NULL)
loadGNN(file)
}
\arguments{
  \item{\dots}{objects to be saved in \code{file} (under the provided
  names if \code{name} was provided). Those objects which are of class
  \code{"gnn_GNN"} are converted with \code{\link{as.raw}()} before they
  are saved.}
  \item{file}{file name; see the underlying \code{\link{save}()}
  and \code{\link{load}()}.}
  \item{name}{\code{\link{character}} (vector) of name(s) under
  which the objects in \code{\dots} are to be saved in \code{file}. If
  \code{\link{NULL}}, the names of the objects provided by \dots are
  taken as \code{name}.}
}
\value{
  \describe{
    \item{saveGNN()}{nothing (generates an \code{.rda} by side-effect).}
    \item{loadGNN()}{the loaded object(s). Those of class \code{"gnn_GNN"}
    are converted with \code{\link{as.keras}()} before they are
    returned; this also applies to a component of a loaded object of class
    \code{\link{list}}.}
  }
}
\author{Marius Hofert}
\seealso{
  See the underlying functions \code{\link{load}()}
  and \code{\link{save}()} (among others).
}
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
file <- tempfile("foo", fileext = ".rda")
GMMN1 <- GMMN()
saveGNN(GMMN1, file = file) # converts GMMN via as.raw()
GMMN2 <- loadGNN(file) # converts loaded object via as.keras()
stopifnot(is.GNN(GMMN2), inherits(GMMN2[["model"]], "keras.engine.training.Model"))
rm(GMMN1, GMMN2) # clean-up
stopifnot(file.remove(file))
}
}
\keyword{manip}
