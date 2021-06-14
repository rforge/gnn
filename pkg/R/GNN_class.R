### Generics ###################################################################

## Note: - print(), str(), summary(), for example, are already generics, so
##         don't need to be defined as such. Also note that dim() is already a
##         generic (even if not directly visible on 'dim').
##       - But note that the corresponding methods need to have the same
##         or at least compatible arguments with the (already defined) generics;
##         otherwise "checking S3 generic/method consistency ... WARNING" appears.
##       - For defining a generic without overwriting already defined generics,
##         see https://gist.github.com/datalove/88f5a24758b2d8356e32

## Generic for checking whether an object is of class "gnn_GNN"
is.GNN <- function(x) UseMethod("is.GNN")

## Generic for printing objects of class "human_proc_time"
print.as.human <- function(x) UseMethod("print.as.human")


### Methods ####################################################################

##' @title Print Method for Objects of Class "gnn_GNN"
##' @param x object of class "gnn_GNN"
##' @return return value of the print method for objects of class "list"
##' @author Marius Hofert
##' @note - A frequent problem when modifying print using print is the error
##'         "Error: C stack usage  7971664 is too close to the limit" due
##'         to recursive calling of the print method for objects of this class
##'         => unclass
##'       - '...' are required because of print generic of this form
print.gnn_GNN <- function(x, ...)
{
    stopifnot(inherits(x, "gnn_GNN"))
    res <- x
    res[["model"]] <- noquote(paste0("object of class \"",class(res[["model"]])[1],"\""))
    if(length(res[["loss"]]) > 7) {
        dgts <- getOption("digits")
        fmt <- paste0("%.",dgts,"f")
        res[["loss"]] <- noquote(paste(paste(sprintf(fmt, res[["loss"]][1:7]), collapse = " "), "..."))
    }
    dm <- dim(res[["prior"]])
    res[["prior"]] <- noquote(paste0("(",dm[1],", ",dm[2],")-matrix of prior samples"))
    res <- unclass(res) # see 'note' above
    print(res)
}

##' @title Str Method for Objects of Class "gnn_GNN"
##' @param object object of class "gnn_GNN"
##' @return return value of the str method for objects of class "list" (nothing!)
##' @author Marius Hofert
##' @note Since str() doesn't return anything, we cannot first call str() and then
##'       modify the output.
str.gnn_GNN <- function(object, ...)
{
    stopifnot(inherits(object, "gnn_GNN"))
    nms <- names(object)
    cls <- class(object[["model"]])[1]
    ## Model part; see calls utils:::str.default()
    cat("List of ",length(object),"\n $ model", # "List of ..." string
        rep(" ", max(max(nchar(nms)) - 5, 0)), # nchar("model") = 5
        ": object of class \"",cls,"\"\n", sep = "")
    ## str() of rest of the list
    str(object[nms != "model"], no.list = TRUE) # omit "List of ..." string
}

##' @title Summary Method for Objects of Class "gnn_GNN"
##' @param object object of class "gnn_GNN"
##' @return return value of the summary method for objects of class "list"
##' @author Marius Hofert
summary.gnn_GNN <- function(object, ...)
{
    stopifnot(inherits(object, "gnn_GNN"))
    smm <- summary(unclass(object)) # calls summary.default() on the list unclass(object)
    smm[,"Class"] <- c(smm["model", "Class"], # fix "Class" column
                       sapply(object[names(object) != "model"], function(x) class(x)[1]))
    smm
}

##' @title Dim Method for Objects of Class "gnn_GNN"
##' @param x object of class "gnn_GNN"
##' @return dimension slot, a vector
##' @author Marius Hofert
dim.gnn_GNN <- function(x)
{
    stopifnot(inherits(x, "gnn_GNN"))
    x[["dim"]]
}

##' @title Check for Being an Object of Class "gnn_GNN"
##' @param x R object
##' @return logical indicating whether 'x' is of class "gnn_GNN"
##' @author Marius Hofert
is.GNN.gnn_GNN <- function(x) inherits(x, "gnn_GNN")

##' @title Check for Being a List of Objects of Class "gnn_GNN"
##' @param x R object
##' @return logical indicating whether 'x' is a list of objects of class "gnn_GNN"
##' @author Marius Hofert
is.GNN.list <- function(x)
{
    if(inherits(x, "list")) {
        sapply(x, function(x.) inherits(x., "gnn_GNN"))
    } else stop("'x' must be a list")
}
