\name{rm_ext}
\alias{rm_ext}
\title{Remove a File Extension}
\description{
  Fixes the removal of file extensions of \code{file_path_sans_ext()}
  in the case where file names contain digits after the last dot
  (which is often used to incorporate numeric numbers into file names).
}
\usage{
rm_ext(x)
}
\arguments{
  \item{x}{file name(s) with extension(s) to be stripped off.}
}
\value{
  The file name without its extension (if the file name had an extension).
}
\author{Marius Hofert}
\examples{
myfilepath1 <- "/myusername/my_filename_with_dots_0.25_0.50_0.75.rda"
myfilepath2 <- "/myusername/my_filename_with_dots_0.25_0.50_0.75"
myfilepath3 <- "/myusername/my_filename_with_dots_0.25_0.50_0.75."
myfilepath4 <- "/myusername/my_filename_with_dots_0.25_0.50_0.75._"
myfilepath5 <- "/myusername/my_filename_with_dots_0.25_0.50_0.75._*.rda"
library(tools)
file_path_sans_ext(myfilepath2) # fails (only case)

stopifnot(rm_ext(myfilepath1) == file_path_sans_ext(myfilepath1))
stopifnot(rm_ext(myfilepath2) == myfilepath2)
stopifnot(rm_ext(myfilepath3) == file_path_sans_ext(myfilepath3))
stopifnot(rm_ext(myfilepath4) == file_path_sans_ext(myfilepath4))
stopifnot(rm_ext(myfilepath5) == file_path_sans_ext(myfilepath5))
}
\keyword{utilities}
