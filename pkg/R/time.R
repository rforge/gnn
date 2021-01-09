### Human-readable time measurement ############################################

##' @title Convert Objects of Class "proc_time" to Human-Readable Format
##' @param x object of class "proc_time"
##' @param fmt sprintf() format string
##' @return character(3) giving the user, system and elapsed times in
##'         human-readable format
##' @author Marius Hofert
as.human <- function(x, fmt = "%.2f")
{
    if(!inherits(x, "proc_time"))
        stop("'x' must be an object of class \"proc_time\"")
    x <- x[1:3] # grab out first three components only
    res <- sapply(x, function(t) {
        if(t < 60) {
            sprintf(paste0(fmt,"s"), t)
        } else if(t < 3600) {
            sprintf(paste0(fmt,"min"), t/60)
        } else {
            sprintf(paste0(fmt,"h"), t/3600)
        }
    })
    names(res) <- c("user", "system", "elapsed")
    res
}

##' @title Human-Readable Time Measurement
##' @param expr see ?system.time
##' @param print logical indicating whether to print the result;
##'        either way, it is returned (invisibly if print = TRUE)
##' @param ... additional arguments passed to the underlying as.human()
##' @return see ?as.human
##' @author Marius Hofert
human.time <- function(expr, print = TRUE, ...)
{
    proc <- system.time(expr)
    hproc <- as.human(proc, ...)
    if(print) {
        print(hproc, quote = FALSE)
        invisible(hproc)
    } else {
        hproc
    }
}

##' @title Method for Objects of Class "gnn_GNN"
##' @param x object of class "gnn_GNN"
##' @param human logical indicating whether to convert
##'        times into human-readable format
##' @param ... additional arguments passed to the underlying as.human()
##' @return object of class "gnn_proc_time"
##' @author Marius Hofert
time.gnn_GNN <- function(x, human = FALSE, ...)
{
    if(!inherits(x, "gnn_GNN"))
        stop("'x' must be an object of class \"gnn_GNN\"")
    proc <- x[["time"]] # object of class "proc_time"
    if(human) {
        res <- as.human(proc, ...) # only a character(3) but we define a plot method to remove ""
    } else {
        res <- proc[1:3] # loses class "proc_time" and is now numeric(3)
        names(res) <- c("user", "system", "elapsed")
    }
    structure(res, class = "gnn_proc_time") # define a class for which we then define a print method
}

##' @title Print Method for Objects of Class "gnn_proc_time"
##' @param x object of class "gnn_proc_time"
##' @param ... not used; for compatibility with 'print' generic
##' @return x (invisibly)
##' @author Marius Hofert
print.gnn_proc_time <- function(x, ...)
{
    stopifnot(inherits(x, "gnn_proc_time"))
    y <- unclass(x) # remove class attribute (to call correct print method and don't enter loop)
    if(is.character(y)) {
        print(y, quote = FALSE)
    } else { # numeric
        print(y)
    }
    invisible(x)
}
