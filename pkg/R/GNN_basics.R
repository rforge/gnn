### GNN basic functions ########################################################

##' @title Check for Class "gnn_GNN"
##' @param x single object or list of such
##' @return logical indicating if GNN
##' @author Marius Hofert
##' @note a bit smarter than 'just' inherits(x, "gnn_GNN") which is
##'       still useful for exact checks
is.GNN <- function(x)
{
    is.gnn <- inherits(x, "gnn_GNN")
    if(is.gnn) {
        TRUE
    } else if(is.list(x)) { # could still be a list of such
        sapply(x, function(x.) inherits(x., "gnn_GNN"))
    } else { # not even a list
        FALSE
    }
}


### GNN basic generics #########################################################

## Note: - print(), str(), summary() are already generics, so don't need to be
##         defined as such. Also note that dim() is already a generic (even if
##         not directly visible on 'dim').
##       - But note that the corresponding methods need to have the same
##         or at least compatible arguments with the (already defined) generics;
##         otherwise "checking S3 generic/method consistency ... WARNING" appears.
##       - For defining a generic without overwriting already defined generics,
##         see https://gist.github.com/datalove/88f5a24758b2d8356e32


### GNN basic methods ##########################################################

##' @title Print Method for Objects of Class "gnn_GNN"
##' @param x object of class "gnn_GNN"
##' @return return value of the print method for objects of class "list"
##' @author Marius Hofert
##' @note Just replace component 'model' by our own (to avoid keras print method)
print.gnn_GNN <- function(x, ...)
{
    stopifnot(inherits(x, "gnn_GNN"))
    cls <- class(x[["model"]])[1]
    print(c("model" = noquote(paste0("object of class \"",cls,"\"")), # new 'model' component
            x[names(x) != "model"]))
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
    summ <- summary(unclass(object)) # calls summary.default() on the list unclass(object)
    summ[,"Class"] <- c(summ["model", "Class"], sapply(object[names(object) != "model"], class)) # fix classes
    summ
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
