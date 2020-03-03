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


### system.time() with human-readable output ###################################

##' @title system.time() with Human-Readable Output
##' @param ... arguments passed to the underlying system.time()
##' @param digits see ?round
##' @return timings in human-readable format
##' @author Marius Hofert
human_time <- function(..., digits = 2) {
    toHuman <- function(t) {
        if(!is.numeric(t)) return(character(0))
        if(t < 60) {
            paste0(round(t, digits = digits),"s")
        } else if(t < 3600) {
            paste0(round(t/60, digits = digits),"min")
        } else {
            paste0(round(t/3600, digits = digits),"h")
        }
    }
    tm <- system.time(...) # note: has NA for Windows, see ?proc.time
    res <- sapply(tm[1:3], toHuman)
    names(res) <- c("user", "system", "elapsed")
    noquote(res)
}
