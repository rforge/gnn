### Auxiliaries ################################################################

##' @title Remove an Extension from a File Path By Only Cutting off after the
##'        last Period
##' @param x character strings with file names and (potentially) extensions
##'        to be stripped off
##' @return file name without extension (but only if extension doesn't start
##'         with a digit (because it's then part of the file name that was
##'         provided without extension in this case)
##' @author Marius Hofert
##' @note See https://stackoverflow.com/questions/57182339/how-to-strip-off-a-file-ending-but-only-when-it-starts-with-a-non-digit-a-rege
rm_ext <- function(x) sapply(x, function(x.) sub("\\.(?:[^0-9.][^.]*)?$", "", x.))
