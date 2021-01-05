### Catching warnings and errors ###############################################

##' @title Catching Results, Warnings and Errors Simultaneously
##' @param expr assignment or function evaluation
##' @return list with components:
##'         'value'  : value of expr or simpleError
##'	    'warning': simpleWarning or NULL
##'         'error'  : simpleError or NULL
##' @author Marius Hofert (see simsalapar)
##' @note https://stat.ethz.ch/pipermail/r-help/2010-December/262626.html
catch <- function(expr)
{
    W <- NULL
    w.handler <- function(w) { # warning handler
	W <<- w
	invokeRestart("muffleWarning")
    }
    res <- list(value = withCallingHandlers(tryCatch(expr, error = function(e) e),
                                            warning = w.handler), warning = W)
    is.err <- is(val <- res$value, "simpleError") # logical indicating an error
    list(value   = if(is.err) NULL else val, # value (or NULL in case of error)
	 warning = res$warning, # warning (or NULL)
	 error   = if(is.err) val else NULL) # error (or NULL if okay)
}


### Simple and (re)strict(ive) test of whether TensorFlow is available #########

##' @title Test whether TensorFlow is Available
##' @return boolean indicating whether TensorFlow is found
##' @author Marius Hofert
##' @note See https://stackoverflow.com/questions/38549253/how-to-find-which-version-of-tensorflow-is-installed-in-my-system
TensorFlow_available <- function() {
    if(Sys.info()[["sysname"]] == "Windows") {
        warning("'TensorFlow_available()' does not work on Windows. Will return FALSE.")
        return(FALSE)
    }
    TFfound <- catch(system("pip list | grep tensorflow", ignore.stdout = TRUE))
    is.null(TFfound$error) && is.null(TFfound$warning) && (TFfound$value == 0)
}


### Remove extension of a file path ############################################

##' @title Fixed Version of tools::file_path_sans_ext()
##' @param x character strings with file names and (potentially) extensions
##'        to be stripped off
##' @return file name without extension
##' @author Marius Hofert
##' @note Idea: Find the last dot in the file name. If there is a number
##'             thereafter, call our fix of tools::file_path_sans_ext(),
##'             otherwise call tools::file_path_sans_ext().
rm_ext <- function(x)
    sapply(x, function(x.) {
        splt <- strsplit(x., split = "")[[1]]
        dots <- splt == "."
        whch <- which(dots)
        lwhch <- length(whch)
        if(lwhch == 0 || dots[length(splt)]) # x. contains no dots or ends in a dot
            return(file_path_sans_ext(x.))
        ## Now we have at least one dot and the string does not end in a dot.
        ## Figure out the element after the last dot
        ind <- whch[lwhch] # index of last dot
        char.after.last.dot <- splt[ind + 1] # element after last dot
        if(grepl("^[[:digit:]]", x = char.after.last.dot)) { # see https://stackoverflow.com/questions/13638377/test-for-numeric-elements-in-a-character-string
            sub("\\.(?:[^0-9.][^.]*)?$", "", x.) # see https://stackoverflow.com/questions/57182339/how-to-strip-off-a-file-ending-but-only-when-it-starts-with-a-non-digit-a-rege
        } else file_path_sans_ext(x.)
    }, USE.NAMES = FALSE)
