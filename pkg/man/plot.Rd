\name{plot}
\alias{plot_loss}
\title{Functions for Plotting}
\description{
  Functions for plotting.
}
\usage{
plot_loss(x, type = "l", xlab = "Epoch", ylab = "Loss", y2lab = NULL, ...)
}
\arguments{
  \item{x}{trained object of class \code{"gnn_GNN"}
    whose loss function (loss per epoch of training) is to be plotted.}
  \item{type}{line type; see \code{\link{plot}()}.}
  \item{xlab}{x-axis label; see \code{\link{plot}()}.}
  \item{ylab}{y-axis label; see \code{\link{plot}()}.}
  \item{y2lab}{secondary y-axis label.}
  \item{\dots}{additional arguments passed to the underlying
    \code{\link{plot}()}.}
}
\value{
  Invisibly the losses per epoch.
}
\author{Marius Hofert}
\seealso{
  \code{\link{fitGNN}()}.
}
\keyword{hplot}
