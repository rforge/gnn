\name{rda}
\alias{exists_rda}
\alias{read_rda}
\alias{save_rda}
\alias{rename_rda}
\title{Check Existence, Read, Save and Rename .rda Files and their Objects}
\description{
  Check Existence, Read, Save and Rename \code{.rda} Files and their Objects.
}
\usage{
exists_rda(file, objnames, package = NULL)
read_rda(objnames, file, package = NULL)
save_rda(..., file, names)
rename_rda(oldname, oldfile = paste0(oldname, collapse = "_"),
           newname, newfile = paste0(newname, collapse = "_", ".rda"),
	   package = NULL)
}
\arguments{
  \item{file}{
    \describe{
      \item{exists_rda()}{\code{\link{character}} string (with or
	without ending \code{.rda}) specifying the name of the file
	to check existence of (if \code{pacakge = NULL}) or in (otherwise).}
      \item{read_rda()}{\code{\link{character}} string (with or without ending
	\code{.rda}) specifying the file to read from.}
      \item{save_rda()}{\code{\link{character}} string (with or without ending
	\code{.rda}) specifying the file to save to.}
    }
  }
  \item{objnames}{
    \describe{
      \item{exists_rda()}{\code{\link{character}} vector of names of objects to
	be checked for existence.}
      \item{read_rda()}{\code{\link{character}} vector of names of
	objects to be read. If not provided, a name is constructed
	from \code{file}.}
    }
  }
  \item{package}{
    \describe{
      \item{exists_rda()}{package name in which to check or \code{\link{NULL}}
	(the default) in which case the current working directory is checked.}
      \item{read_rda(), rename_rda()}{package name from which to load
	the objects or \code{\link{NULL}} (the default) in which case
	the current working directory is searched.}
    }
  }
  \item{\dots}{any number of \R objects.}
  \item{names}{\code{\link{character}} \code{\link{vector}} of names
    under which the objects in \code{...} are to be saved in
    the file \code{file}.}
  \item{oldname}{\code{\link{character}} string specifying the object to
    be read.}
  \item{oldfile}{file name (with or without ending \code{.rda})
    specifying from which the object named \code{oldname} is read.}
  \item{newname}{\code{\link{character}} string specifying the new name
    under which the object is to be saved.}
  \item{newfile}{file name (with ending \code{.rda}) specifying
    where the object named \code{oldname} is saved under the name
    \code{newname}.}
}
\value{
  \describe{
    \item{exists_rda()}{\code{\link{logical}} indicating whether the
      \code{.rda} file \code{file} exists (if \code{objnames} is not
      provided) or whether the objects with names \code{objnames}
      exist inside \code{file} (if \code{objnames} is provided).}
    \item{read_rda()}{the object read from the \code{.rda}.}
    \item{save_rda()}{nothing (generated an \code{.rda} by side-effect).}
    \item{rename_rda()}{nothing (generated an \code{.rda} by side-effect).}
  }
}
\author{Marius Hofert}
\seealso{
  See the underlying functions \code{\link{load}()},
  \code{\link{data}()} and \code{\link{save}()} (among others).
}
\keyword{manip}
