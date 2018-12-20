\name{save_rda}
\alias{save_rda}
\title{Save R Objects under Given Names}
\description{
  Save \R objects under given names via \code{\link{save}()}.
}
\usage{
save_rda(x, names = names(x),
         file = paste0(paste0(names(x), collapse = "_"), ".rda"), ...)
}
\arguments{
  \item{x}{\code{\link{list}} of \R objects to be saved in \code{file}
    under the names contained in \code{names}.}
  \item{names}{\code{\link{character}} \code{\link{vector}} containing
    the names for the components of \code{x}.}
  \item{file}{base name (\code{\link{character}} string) of the file
    under which to save \code{x}.}
  \item{\dots}{additional arguments passed to the underlying
    \code{\link{save}()}.}
}
\value{
  see \code{\link{save}()}
}
\author{Marius Hofert}
\seealso{\code{\link{dataset_exists}()}, \code{\link{read_dataset}()}}
\keyword{manip}
