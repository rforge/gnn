\name{plot}
\alias{plot}
\alias{plot.gnn_GNN}
\title{Functions for Plotting}
\description{
  Functions for plotting.
}
\usage{
\method{plot}{gnn_GNN}(x, plot.type = "loss", type = "l", xlab = "Epoch",
    ylab = paste0("Loss (",x[["loss.type"]],")"),
    y2lab = NULL, ...)
}
\arguments{
  \item{x}{trained object of class \code{"gnn_GNN"}
    whose loss function (loss per epoch of training) is to be plotted.}
  \item{plot.type}{\code{\link{character}()} indicating the type of plot.}
  \item{type}{line type; see \code{\link{plot}()}.}
  \item{xlab}{x-axis label; see \code{\link{plot}()}.}
  \item{ylab}{y-axis label; see \code{\link{plot}()}.}
  \item{y2lab}{secondary y-axis label.}
  \item{\dots}{additional arguments passed to the underlying
    \code{\link{plot}()}.}
}
\value{
  Invisible (plot by side-effect).
}
\author{Marius Hofert}
\seealso{
  \code{\link{fitGNN}()}.
}
\keyword{hplot}
