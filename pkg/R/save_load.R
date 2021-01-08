### Saving and loading of objects (with conversion to raw/keras models) ########

##' @title Saving Objects Including GNNs in a Given .rda File
##' @param ... objects to be saved in 'file' under names specified by 'name'.
##'        Those of class "gnn_GNN" are converted with as.raw().
##' @param file character string (with or without extension '.rda') specifying
##'        the file to save to
##' @param name character vector of names under which the objects are saved
##'        in 'file' (defaults to the names of the arguments provided by '...')
##' @return nothing (generates an .rda by side-effect)
##' @author Marius Hofert
##' @note as save(), saveGNN() also overwrites files
saveGNN <- function(..., file, name = NULL)
{
    ## Basics
    stopifnot(is.character(file), length(file) == 1)
    args <- list(...)
    len <- length(args)
    ## If 'name' was not provided determine it
    if(is.null(name)) {
        nms <- deparse(substitute(list(...))) # get names of provided arguments
        nms <- substring(nms, first = 6, last = nchar(nms) - 1) # strip away "list(" and ")"
        name <- unlist(strsplit(nms, split = ", "))
    }
    ## Iterate over all objects and rename them. If of class "gnn_GNN", call as.raw()
    if(length(name) != len)
        stop("Length of 'name' must be equal to the number of objects provided by '...'")
    for(i in seq_len(len)) {
        if(inherits(args[[i]], "gnn_GNN")) # order import (otherwise the saved $model component is <pointer: 0x0>)
            args[[i]] <- as.raw(args[[i]]) # first convert...
        assign(name[i], value = args[[i]]) # ... then name the objects in 'args' as specified by 'name'
    }
    ## Save
    save(list = name, file = file) # save R objects in 'file' under the provided 'name'
}

##' @title Reading Objects Including GNNs from a Given .rda File
##' @param file character string (with or without extension .rda) specifying
##'        the file to read from
##' @return the read object(s); those of class "gnn_GNN" (possibly contained in a list)
##'         are converted with as.keras().
##' @author Marius Hofert
##' @note as readRDS(), behaves more functional in that it returns an object
loadGNN <- function(file)
{
    stopifnot(is.character(file))
    file <- paste0(rm_ext(file),".rda") # file with extension
    ## Create a temporary environment to load the objects into in order to
    ## modify them and return them (without convoluting .GlobalEnv
    myenvir <- new.env()
    nms <- load(file, envir = myenvir) # load objects into myenvir
    len <- length(nms)
    objs <- if(len == 1) { # get read objects by name
                get(nms, envir = myenvir)
            } else if(len > 1) {
                mget(nms, envir = myenvir)
            } else stop("No objects found in 'file'")
    ## Modify them
    if(inherits(objs, "gnn_GNN")) {
        objs <- as.keras(objs)
    } else if(inherits(objs, "list")) {
        for(i in seq_len(length(objs)))
            if(inherits(objs[[i]], "gnn_GNN")) objs[[i]] <- as.keras(objs[[i]])
    }
    ## Clean-up and return
    rm(myenvir) # clean-up
    objs # return
}
