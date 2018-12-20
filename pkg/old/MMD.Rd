\name{MMD}
\alias{MMD}
\title{Maximum Mean Discrepancy Loss Function}
\description{
  Implementation of the maximum mean discrepancy (MMD) loss
  function to measure statistical discrepancy between two datasets.
}
\usage{
MMD(x, y, bw = c(0.001, 0.01, 0.15, 0.25, 0.50, 0.75))
}
\arguments{
  \item{x}{2d tensor with shape (batch size, dimension of input dataset).}
  \item{y}{2d tensor with shape (batch size, dimension of input dataset).}
  \item{bw}{\code{\link{numeric}} (vector) containing the bandwidth
    parameter(s) considered; the default seems suitable for
    copula samples as they naturally lie in \eqn{[0,1]^d}.}
}
\value{
  \code{MMD()} returns a 0d tensor containing the maximum mean
  discrepancy (MMD) loss.
}
\author{Marius Hofert and Avinash Prasad}
\references{
  Gretton, A., Borgwardt, K. M., Rasch, M., \enc{Schölkopf}{Schoelkopf}, B.
  and Smola, A. J. (2007).
  A kernel method for the two-sample-problem.
  \emph{Advances in Neural Information Processing Systems}, \bold{19}, 513--520.
  See https://arxiv.org/abs/0805.2368
}
\seealso{\code{\link{GMMN_model}()}, \code{\link{RL}()}}
\keyword{univar}
