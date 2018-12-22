### Auxiliary functions for checking existence, loading and saving data ########

##' @title Check the Existence of a Dataset(s) in a Package
##' @param x character vector with names of datasets (.rda) to be checked for
##'        their existence in 'package'
##' @param package character string indicating the package in which to check
##'        the existence of 'x'
##' @param envir argument of data(); required to avoid note on check
##' @param ... additional arguments passed to the underlying data()
##' @return logical vector
##' @author Marius Hofert
dataset_exists <- function(x, package = packageName(), envir = .GlobalEnv, ...)
    x %in% data(package = package, envir = envir, ...)[["results"]][,"Item"]

##' @title Load and Return Datasets from a Package
##' @param x character vector with names of datasets (.rda) to be loaded from
##'        'package'
##' @param rm logical indicating whether the loaded R objects are to be removed
##'        from the global environment
##' @param package character string indicating the package from which to load
##'        the R object(s)
##' @param envir argument of data() providing the environment in which the
##'        R objects are loaded; required to avoid note on check
##' @param ... additional arguments passed to the underlying data()
##' @return (a list containing) the loaded R object(s)
##' @author Marius Hofert
##' @note There could be a 'read_rda' at some point (reading from a file)
read_dataset <- function(x, rm = TRUE, package = packageName(), envir = .GlobalEnv, ...)
{
    ii <- dataset_exists(x, package = package, ...)
    if(length(ii) == 0)
        stop(paste0("No data set specified in 'x' was found in package ",dQuote(package)))
    x. <- x[ii] # which do exist?
    data(list = x., package = package, envir = envir, ...) # load existing datasets
    ## Note: get(<vector>) = get(<1st element only>)
    res <- if(length(x.) > 1) lapply(x., get) else get(x.) # return the loaded data object(s)
    if(rm) rm(list = x., envir = .GlobalEnv) # remove objects from global environment
    res
}

##' @title Save R Objects under Given Names
##' @param x (list of) R objects to be saved to 'file'
##' @param names character vector containing the names for the components of 'x'
##' @param file base name (character string) of the file under which to save 'x'
##' @param ... additional arguments passed to the underlying save()
##' @return see ?save
##' @author Marius Hofert
save_rda <- function(x, names = names(x),
                     file = paste0(paste0(names(x), collapse = "_"), ".rda"), ...)
{
    stopifnot(is.character(file), !is.null(names))
    if(is.list(x) && !is.data.frame(x)) { # create objects with names 'names' containing the objects in x
        for(i in seq_len(length(x))) # note: fails with sapply()
            assign(names[i], value = x[[i]])
    } else assign(names, value = x)
    save(list = names, file = file, ...) # save R object(s) 'names' in 'file'
}

## Auxiliary functions for converting keras model weights to R objects and vice versa

##' @title Convert Keras model weights to R objects
##' @param model A Keras model
##' @return R object containing 'model' weights
##' @author Avinash Prasad
##' @note In the same vein as Keras function serialize_model() which serializes Keras models 
##' (rather than model weights)  to R objects.
serialize_weights<-function(model){
  # Ensure model provided is a Keras model
  stopifnot(inherits(model, "keras.engine.training.Model"))
  ## create a temporary hdf5 file to which we write the Keras model weights
  tmp <- tempfile(pattern = "file", fileext = ".h5")  
  on.exit(unlink(tmp), add = TRUE)
  save_model_weights_hdf5(model, tmp)
  
  # Convert saved temp hdf5 file back to an R object
  readBin(tmp, what = "raw", n = file.size(tmp))
}
##' @title Load model weights into a keras model from a R object
##' @param model A Keras model
##' @param model.weights R object containing model weights
##' @return Keras model with 'model.weights' loaded into 'model'.
##' @author Avinash Prasad
##' @note In the same vein as Keras function unserialize_model() which loads R objects
##' into keras models.
unserialize_weights<-function(model,model.weights){
  # Ensure model provided is a Keras model
  stopifnot(inherits(model, "keras.engine.training.Model"))
  ## Create a temp hdf5 file in which we place the model weights provided by the R object 
  tmp <- tempfile(pattern = "file", fileext = ".h5")  
  on.exit(unlink(tmp), add = TRUE)
  writeBin(model.weights, tmp)
  ## Load the model weights back into the model
  load_model_weights_hdf5(model, tmp)
}
