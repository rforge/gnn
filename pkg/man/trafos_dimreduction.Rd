\name{trafos_dimreduction}
\alias{PCA_trafo}
\title{Dimension-Reduction Transformations for Training or Sampling}
\description{
  Dimension-reduction transformations applied to an input data matrix.
  Currently on the principal component transformation and its inverse.
}
\usage{
PCA_trafo(x, mu, Gamma, inverse = FALSE, ...)
}
\arguments{
  \item{x}{\eqn{(n, d)}-matrix of data (typically before training or
    after sampling). If \code{inverse = FALSE}, then, conceptually,
    an \eqn{(n, d)}-matrix with \eqn{1\le k \le d}{1 <= k <= d},
    where \eqn{d} is the dimension of the original data whose dimension
    was reduced to \eqn{k}.}
  \item{mu}{if \code{inverse = TRUE}, a \eqn{d}-vector
    of centers, where \eqn{d} is the dimension to transform \code{x}
    to.}
  \item{Gamma}{if \code{inverse = TRUE}, a \eqn{(d, k)}-matrix with
    \eqn{k} at least as large as \code{ncol(x)} containing the
    \eqn{k} orthonormal eigenvectors of a covariance matrix sorted
    in decreasing order of their eigenvalues; in other words,
    the columns of \code{Gamma} contain principal axes or loadings.
    If a matrix with \eqn{k} greater than \code{ncol(x)} is provided,
    only the first \eqn{k}-many are considered.}
  \item{inverse}{\code{\link{logical}} indicating whether the inverse
    transformation of the principal component transformation is applied.}
  \item{\dots}{additional arguments passed to the underlying
    \code{\link{prcomp}()}.}
}
\value{
  If \code{inverse = TRUE}, the transformed data whose rows contain
  \eqn{\bm{X} = \bm{\mu} + \Gamma \bm{Y}}{X = mu + Gamma Y}, where
  \eqn{Y} is one row of \code{x}. See the details below for the
  notation.

  If \code{inverse = FALSE}, a \code{\link{list}} containing:
  \describe{
    \item{\code{PCs}:}{\eqn{(n, d)}-matrix of principal components.}
    \item{\code{cumvar}:}{cumulative variances; the \eqn{j}th entry
      provides the fraction of the explained variance of the first
      \eqn{j} principal components.}
    \item{\code{sd}:}{sample standard deviations of the transformed data.}
    \item{\code{lambda}:}{eigenvalues of \code{cov(x)}.}
    \item{\code{mu}:}{\eqn{d}-vector of centers of \code{x} (see also
      above) typically provided to \code{PCA_trafo(, inverse = TRUE)}.}
    \item{\code{Gamma}:}{\eqn{(d, d)}-matrix of principal axes (see also
      above) typically provided to \code{PCA_trafo(, inverse = TRUE)}.}
  }
}
\details{
  Conceptually, the principal component transformation transforms a
  vector \eqn{\bm{X}}{X} to a vector \eqn{\bm{Y}}{Y} where
  \eqn{\bm{Y} = \Gamma^T(\bm{X}-\bm{\mu})}{Y = Gamma^T (X - mu)},
  where \eqn{\bm{\mu}}{\mu} is the mean vector of \eqn{\bm{X}}{X}
  and \eqn{\Gamma}{Gamma} is the \eqn{(d, d)}-matrix whose
  columns contains the orthonormal eigenvectors of \code{cov(X)}.

  The corresponding (conceptual) inverse transformation is
  \eqn{\bm{X} = \bm{\mu} + \Gamma \bm{Y}}{X = mu + Gamma Y}.

  See McNeil et al. (2015, Section 6.4.5).
}
\author{Marius Hofert}
\references{
   McNeil, A. J., Frey, R., and Embrechts, P. (2015).
   \emph{Quantitative Risk Management: Concepts, Techniques, Tools}.
   Princeton University Press.
}
\examples{
## Generate data
library(copula)
set.seed(271)
X <- qt(rCopula(1000, gumbelCopula(2, dim = 10)), df = 3.5)
pairs(X, gap = 0, pch = ".")

## Principal component transformation
PCA <- PCA_trafo(X)
Y <- PCA$PCs
PCA$cumvar[3] # fraction of variance explained by the first 3 principal components
which.max(PCA$cumvar > 0.9) # number of principal components it takes to explain 90%

## Biplot (plot of the first two principal components = data transformed with
## the first two principal axes)
plot(Y[,1:2])

## Transform back and compare
X. <- PCA_trafo(Y, mu = PCA$mu, Gamma = PCA$Gamma, inverse = TRUE)
stopifnot(all.equal(X., X))

## Note: One typically transforms back with only some of the principal axes
X. <- PCA_trafo(Y[,1:3], mu = PCA$mu, # mu determines the dimension to transform to
                Gamma = PCA$Gamma, # must be of dim. (length(mu), k) for k >= ncol(x)
                inverse = TRUE)
stopifnot(dim(X.) == c(1000, 10))
## Note: We (typically) transform back to the original dimension.
pairs(X., gap = 0, pch = ".") # pairs of back-transformed first three PCs
}
\keyword{manip}
