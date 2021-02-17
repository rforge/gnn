\name{plot}
\alias{plot}
\alias{plot.gnn_GNN}
\title{Functions for Plotting}
\description{
  Functions for plotting.
}
\usage{
\method{plot}{gnn_GNN}(x, plot.type = c("scatter", "loss"), max.n.samples = NULL,
    type = NULL, xlab = NULL, ylab = NULL,
    y2lab = NULL, labels = "X", pair = NULL, ...)
}
\arguments{
  \item{x}{trained object of class \code{"gnn_GNN"}
    whose loss function (loss per epoch of training) is to be plotted.}
  \item{plot.type}{\code{\link{character}()} indicating the type
    of plot.}
  \item{max.n.samples}{maximal number of samples to be plotted.}
  \item{type}{line type; see \code{\link{plot}()}.}
  \item{xlab}{x-axis label; see \code{\link{plot}()}.}
  \item{ylab}{y-axis label; see \code{\link{plot}()}.}
  \item{y2lab}{secondary y-axis label.}
  \item{labels}{\code{\link{character}()} vector indicating
    the labels to be used; if of length 1, then the base label
    to be used.}
  \item{pair}{\code{\link{numeric}(2)} containing the indices of the
    pair to be plotted.}
  \item{\dots}{additional arguments passed to the underlying
    \code{\link{plot}()}.}
}
\value{
  Plot by side-effect.
}
\author{Marius Hofert}
\seealso{
  \code{\link{fitGNN}()}.
}
\keyword{hplot}
