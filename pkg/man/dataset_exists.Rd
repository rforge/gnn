\name{dataset_exists}
\alias{dataset_exists}
\title{Check the Existence of a Dataset(s) in a Package}
\description{
  Check the existence of dataset(s) in a package without loading them.
}
\usage{
dataset_exists(x, package = packageName(), envir = .GlobalEnv, ...)
}
\arguments{
  \item{x}{\code{\link{character}} \code{\link{vector}} with names of
    datasets (\code{.rda}) to be checked for their existence in \code{package}.}
  \item{package}{\code{\link{character}} string indicating the package
    in which to check the existence of \code{x}.}
  \item{envir}{argument of the underlying \code{\link{data}()}; see
    \dQuote{Good practice} on \code{\link{data}()} for why this argument is
    provided.}
  \item{\dots}{additional arguments passed to the underlying
    \code{\link{data}()}.}
}
\value{
  \code{\link{logical}} \code{\link{vector}} indicating whether
  \code{package} contains datasets \code{x}.
}
\author{Marius Hofert}
\seealso{\code{\link{read_dataset}()}, \code{\link{save_rda}()}}
\keyword{manip}
