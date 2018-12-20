\name{read_dataset}
\alias{read_dataset}
\title{Load and Return Datasets from a Package}
\description{
  Load and return datasets from a package via \code{\link{data}()}.
}
\usage{
read_dataset(x, rm = TRUE, package = packageName(), envir = .GlobalEnv, ...)
}
\arguments{
  \item{x}{\code{\link{character}} \code{\link{vector}} with names of
    datasets (\code{.rda}) to be loaded from \code{package}.}
  \item{rm}{\code{\link{logical}} indicating whether the loaded \R
    objects are to be removed from the global environment.}
  \item{package}{\code{\link{character}} string indicating the package
    from which to load the \R objects \code{x}.}
  \item{envir}{argument of the underlying \code{\link{data}()};
    environment in which the \R objects are loaded.}
  \item{\dots}{additional arguments passed to the underlying
    \code{\link{data}()}.}
}
\value{
  (a \code{\link{list}} containing) the loaded \R object(s)
}
\author{Marius Hofert}
\seealso{\code{\link{dataset_exists}()}, \code{\link{save_rda}()}}
\keyword{manip}
