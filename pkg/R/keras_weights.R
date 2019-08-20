### Auxiliary functions for converting Keras model weights to R objects and vice versa

##' @title Convert Keras Model Weights to R Objects
##' @param model A Keras model
##' @return R object containing 'model' weights
##' @author Avinash Prasad
##' @note Similar to keras::serialize_model() which serializes Keras models
##'       (rather than model weights) to R objects.
serialize_weights <- function(model)
{
    stopifnot(inherits(model, "keras.engine.training.Model"))
    tmp <- tempfile(pattern = "file", fileext = ".h5") # create temporary hdf5 file to which we write the Keras model weights
    on.exit(unlink(tmp), add = TRUE)
    save_model_weights_hdf5(model, tmp) # saves model weights to temporary hdf5 file
    readBin(tmp, what = "raw", n = file.size(tmp)) # temporary hdf5 file to R object
}

##' @title Load Model Weights into a Keras Model from an R Object
##' @param model a Keras model
##' @param model.weights R object containing model weights
##' @return Keras model with 'model.weights' loaded into 'model'.
##' @author Avinash Prasad
##' @note In the same vein as Keras function unserialize_model() which
##'       loads R objects into keras models.
unserialize_weights <- function(model, model.weights)
{
    stopifnot(inherits(model, "keras.engine.training.Model"))
    tmp <- tempfile(pattern = "file", fileext = ".h5") # create temporary hdf5 file to which we write the model weights provided by the R object
    on.exit(unlink(tmp), add = TRUE)
    writeBin(model.weights, tmp) # R object to temporary hdf5 file
    load_model_weights_hdf5(model, tmp) # load the model weights back into the model
}
