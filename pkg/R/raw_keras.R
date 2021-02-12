### Auxiliary functions ########################################################

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


### GNN as.raw and as.keras generics ###########################################

## as.raw <- function(gnn) UseMethod("as.raw") # don't define; see https://stackoverflow.com/questions/65472475/how-to-define-an-s3-generic-with-the-same-name-as-a-primitive-function/65472950#65472950
as.keras <- function(x) UseMethod("as.keras")


### GNN as.raw and as.keras methods ############################################

##' @title Convert keras Model to raw
##' @param x object of S3 class "gnn_GNN" to be converted
##' @return object of S3 class "gnn_GNN" with 'model' component converted to
##'         object of class "raw"
##' @author Marius Hofert
##' @note For a VAE, one would need to use serialize_weights() and apply this to
##'       'model', 'encoder' and 'generator'.
as.raw.gnn_GNN <- function(x) # needs 'x' because of generic being already defined (see above)
{
    if(inherits(x[["model"]], "keras.engine.training.Model"))
        x[["model"]] <- serialize_model(x[["model"]]) # serialize component 'model'
    x
}

##' @title Convert raw Model to keras
##' @param x object of S3 class "gnn_GNN" to be converted
##' @return object of S3 class "gnn_GNN" with 'model' component converted to keras
##'         object of class "keras.engine.training.Model"
##' @author Marius Hofert
##' @note - For a VAE, one would need to use unserialize_weights() and apply this to
##'         'model', 'encoder' and 'generator'.
##'       - Use 'x' here for consistency with as.raw.gnn_GNN()
as.keras.gnn_GNN <- function(x)
{
    if(is.raw(x[["model"]]))
        x[["model"]] <- unserialize_model(x[["model"]], # unserialize component 'model'
                                          custom_objects = c(loss = loss, loss_fun = loss))
    ## Note: - This used to be loss = loss (and loss_fun = loss) when run interactively, but
    ##         suddenly stopped to work (2019-10-06).
    ##       - 'loss_fun' has to be the same name as in FNN()
    x
}
