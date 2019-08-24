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


### Converting Keras model weights to R objects and vice versa #################

##' @title Convert Keras Model Weights to Model Weights
##' @param model Keras model
##' @return R object containing the weights of the model 'model'
##' @author Marius Hofert and Avinash Prasad
##' @note Similar to keras::serialize_model() which serializes Keras models
##'       (rather than model weights) to R objects.
serialize_weights <- function(model)
{
    stopifnot(inherits(model, "keras.engine.training.Model"))
    tmp <- tempfile(pattern = "file", fileext = ".h5") # create temporary hdf5 file to which we write the Keras model weights
    on.exit(unlink(tmp), add = TRUE)
    save_model_weights_hdf5(model, filepath = tmp) # saves model weights to temporary hdf5 file
    readBin(tmp, what = "raw", n = file.size(tmp)) # read from temporary hdf5 file to R object
}

##' @title Convert Model Weights to Keras Model Weights
##' @param model Keras model
##' @param model.weights R object containing the model weights
##' @return Keras model with 'model.weights' loaded into 'model'
##' @author Marius Hofert and Avinash Prasad
##' @note In the same vein as the keras function unserialize_model() which
##'       loads R objects into keras models.
unserialize_weights <- function(model, model.weights)
{
    stopifnot(inherits(model, "keras.engine.training.Model"))
    tmp <- tempfile(pattern = "file", fileext = ".h5") # create temporary hdf5 file to which we write the weights 'model.weights' (R object)
    on.exit(unlink(tmp), add = TRUE)
    writeBin(model.weights, tmp) # writes model weights to temporary hdf5 file
    load_model_weights_hdf5(model, filepath = tmp) # load the model weights into 'model' and return the model
}


### Converting GNNs for saving and loading #####################################

##' @title Convert a Callable GNN to a Savable GNN
##' @param gnn GNN object
##' @return the savable (serialized) GNN
##' @author Marius Hofert
to_savable <- function(gnn)
{
    switch(gnn[["type"]],
           "GMMN" = {
               gnn[["model"]] <- serialize_model(gnn[["model"]]) # serialize component 'model'
           },
           "VAE" = {
               gnn[["model"]]     <- serialize_weights(gnn[["model"]])
               gnn[["encoder"]]   <- serialize_weights(gnn[["encoder"]])
               gnn[["generator"]] <- serialize_weights(gnn[["generator"]])
           },
           stop("Wrong 'type'"))
    gnn
}

##' @title Convert a Savable GNN to a Callable GNN
##' @param gnn GNN object
##' @return the callable (unserialized) GNN
##' @author Marius Hofert
to_callable <- function(gnn)
{
    switch(gnn[["type"]],
           "GMMN" = {
               gnn[["model"]] <- unserialize_model(gnn[["model"]], # unserialize component 'model'
                                                   custom_objects = c(loss = loss))
           },
           "VAE" = {
               gnn[["model"]]     <- unserialize_weights(gnn[["model"]],
                                                         model.weights = gnn[["model"]])
               gnn[["encoder"]]   <- unserialize_weights(gnn[["encoder"]],
                                                         model.weights = gnn[["encoder"]])
               gnn[["generator"]] <- unserialize_weights(gnn[["generator"]],
                                                         model.weights = gnn[["generator"]])
           },
           stop("Wrong 'type'"))
    gnn
}
