\name{rm_ext}
\alias{rm_ext}
\title{Remove a File Extension}
\description{
  Remove a file extension in a smarter way than \code{tools::file_path_sans_ext()}.
  If no file ending is provided, \code{rm_ext()} does not interpret the
  last dot as a file extension (which is often used to incorporate
  numeric numbers into file names, like 0.25 or so).
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
\keyword{utilities}
