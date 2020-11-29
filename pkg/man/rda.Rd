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
exists_rda(file, names, package = NULL)
read_rda(file, names, package = NULL)
save_rda(..., file, names = NULL)
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
  \item{names}{
    \describe{
      \item{exists_rda()}{\code{\link{character}} vector of names of objects to
	be checked for existence.}
      \item{read_rda()}{\code{\link{character}} vector of names of
	objects to be read. If not provided, a name is constructed
	from \code{file}.}
      \item{save_rda()}{\code{\link{character}} vector of names under
	which the objects in \code{\dots} are saved in \code{file}. If
        \code{\link{NULL}}, the names of the objects provided by \dots
        are taken as default values.}
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
      \code{.rda} file \code{file} exists (if \code{names} is not
      provided) or whether the objects with names \code{names}
      exist inside \code{file} (if \code{names} is provided).}
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
\examples{
\donttest{ # to avoid win-builder error "Error: Installation of TensorFlow not found"
## Create dummy objects and save them in a temporary file
myobj1 <- 1:10
myobj2 <- 10:1
file1. <- tempfile("foo1", fileext = ".rda") # tempfile() for CRAN checks
file1 <- rm_ext(file1.)
save(myobj1, myobj2, file = file1.)
rm(myobj1, myobj2)

## Testing exists_rda()
## Check the existence of file 'file' with file ending '.rda'
stopifnot(exists_rda(file1.))
## Check the existence of file 'file' without file ending '.rda'
stopifnot(exists_rda(file1))
## Check the existence of 'myobj1' inside 'file' with file ending '.rda'
stopifnot(exists_rda(file1., names = "myobj1"))
## Check the existence of 'myobj1' inside 'file' without file ending '.rda'
stopifnot(exists_rda(file1,  names = "myobj1"))
## Check the existence of a file named 'SP500_const' in 'qrmdata'
stopifnot(exists_rda("SP500_const", package = "qrmdata"))
## Check the existence of 'SP500_const_info' inside 'SP500_const' in 'qrmdata'
stopifnot(exists_rda("SP500_const", names = "SP500_const_info", package = "qrmdata"))

## Testing read_rda()
stopifnot(names(read_rda(c("myobj1", "myobj2"), file = file1.)) == c("myobj1", "myobj2"))
foo <- read_rda("SP500_const", names = "SP500_const_info", package = "qrmdata")
stopifnot(is.data.frame(foo))

## Testing save_rda()
file2. <- tempfile("foo2", fileext = ".rda") # for CRAN
save_rda(foo, file = file2., names = "SP500info2")
bar <- read_rda(file2., names = "SP500info2")
stopifnot(identical(bar, foo))

## Clean-up
stopifnot(file.remove(file1., file2.))
}
}
\keyword{manip}
