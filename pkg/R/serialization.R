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
