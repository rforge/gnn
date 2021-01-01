### GNN basic training functions ###############################################

##' @title Checking Method for Objects of Class "gnn_GNN"
##' @param x object of class "gnn_GNN"
##' @return logical indicating whether 'x' is trained
##' @author Marius Hofert
trained <- function(x)
{
    if(inherits(x, "gnn_GNN")) {
        !is.na(x[["n.train"]])
    } else {
        stop("'x' is not an object of class \"gnn_GNN\"")
    }
}

##' @title Wipe or Set Training Slots
##' @param x object of class "gnn_GNN"
##' @param value if 'x' is trained, FALSE, otherwise numeric(3)
##' @return changed x
##' @author Marius Hofert
"trained<-" <- function(x, value)
{
    stopifnot(inherits(x, "gnn_GNN"))
    if(trained(x)) {
        if(is.logical(value) && (is.na(value) || !value)) { # value = NA or FALSE
            ## Wipe out components
            x[["n.train"]] <- NA_integer_
            x[["batch.size"]] <- NA_integer_
            x[["n.epoch"]] <- NA_integer_
        } else stop("Trained 'x' only allow value = FALSE")
    } else { # not trained
        if(is.numeric(value) && length(value) == 3 && all(value >= 1)) {
            ## Wipe out components
            x[["n.train"]] <- value[1]
            x[["batch.size"]] <- value[2]
            x[["n.epoch"]] <- value[3]
        } else stop("Untrained 'x' only allow value to be a 3-vector of positive integers")
    }
    x
}

##' @title Check for Objects of Class "gnn_GNN" or Lists of Such
##' @param x single object of class "gnn_GNN" or list of such
##' @return logical indicating whether GNN(s) is (are) trained
##' @author Marius Hofert
##' @note similar to is.GNN()
is.trained <- function(x)
{
    is.gnn <- inherits(x, "gnn_GNN")
    if(is.gnn) {
        !is.na(x[["n.train"]])
    } else if(is.list(x)) { # could still be a list of such
        sapply(x, function(x.) {
            inherits(x., "gnn_GNN") && !is.na(x.[["n.train"]])
        })
    } else stop("'x' is not an object of class \"gnn_GNN\" or list of such")
}


### GNN training generics ######################################################

## Generic for main training routine
fitGNN <- function(x, data, ...) UseMethod("fitGNN")

## Generic for training once
fitGNNonce <- function(x, data, ...) UseMethod("fitGNNonce")


### GNN training methods #######################################################

##' @title Training GNNs
##' @param x a single GNN object as returned by constructor(s)
##' @param data (n, d)-matrix containing n d-dimensional observations forming the
##'        training data
##' @param batch.size number of samples per stochastic gradient step
##' @param n.epoch number of epochs (one epoch equals one pass through the complete
##'        training dataset while updating the GNN's parameters)
##' @param prior (n, d)-matrix of prior samples
##' @param verbose see ?keras:::fit.keras.engine.training.Model
##'        and https://stackoverflow.com/questions/47902295/what-is-the-use-of-verbose-in-keras-while-validating-the-model
##'        0 = silent
##'        1 = progress bar
##'        2 = Epoch i/n.epoch
##' @param ... additional arguments passed to the underlying fit();
##'        see ?keras:::fit.keras.engine.training.Model
##' @return trained GNN
##' @author Marius Hofert
fitGNN.gnn_GNN <- function(x, data, batch.size, n.epoch, prior = NULL, verbose = 1, ...)
{
    ## Checks
    if(!is.matrix(data))
        stop("'data' needs to be an (n, d)-matrix containing n d-dimensional training observations.")
    dim.train <- dim(data) # training data dimensions
    stopifnot(inherits(x, "gnn_GNN"),
              1 <= batch.size, batch.size <= dim.train[1], n.epoch >= 1,
              verbose %% 1 == 0, 0 <= verbose, verbose <= 2)
    dim.out <- tail(dim(x), n = 1) # output dimension
    ## Note: for VAEs, dim.out = dim(x)[1] as input and output layer have the same dim
    if(dim.train[2] != dim.out)
        stop("The dimension of the training data does not match the dimension of the output layer of the GNN")

    ## Train and possibly save
    type <- x[["type"]]
    switch(type,
           "GMMN" = {
               if(is.null(prior)) {
                   prior <- rPrior(nrow(data), copula = indepCopula(ncol(data))) # independent N(0,1)
               } else {
                   if(!all(dim(data) == dim(prior)))
                       stop("dim(data) != dim(prior)")
               }
               ## Note:
               ## x = data to be passed through NN as input
               ## y = target/training data to compare against
               x$model %>% fit(x = prior, y = data,
                               batch_size = batch.size, epochs = n.epoch, verbose = verbose, ...) # training
           },
           ## "VAE" = {
           ##     x$model %>% fit(x = data, y = data, # both input and output to the NN are the target/training data
           ##                     batch_size = batch.size, epochs = n.epoch, verbose = verbose, ...)
           ## },
           stop("Wrong 'type'"))

    ## Update slots of 'x'
    x[["n.train"]] <- dim.train[1]
    x[["batch.size"]] <- batch.size
    x[["n.epoch"]] <- n.epoch

    ## Return trained GNN
    x
}

##' @title Training GNNs with Saving and Restoring
##' @param x a single GNN object as returned by constructor(s), or list of such.
##'        All GNNs (even if already trained) are trained and then saved,
##'        unless 'file' is provided and exists in which case 'file' is loaded
##'        and returned.
##' @param data see fitGNN.gnn_GNN()
##' @param batch.size see fitGNN.gnn_GNN()
##' @param n.epoch see fitGNN.gnn_GNN()
##' @param prior see fitGNN.gnn_GNN()
##' @param verbose see fitGNN.gnn_GNN()
##' @param file NULL or a file name in which case the trained GNN is saved in the
##'        provided file. If called again and the file exists, no training is done
##'        but the trained object loaded from the file.
##' @param ... see fitGNN.gnn_GNN()
##' @return see fitGNN.gnn_GNN()
##' @author Marius Hofert
fitGNNonce.gnn_GNN <- function(x, data, batch.size, n.epoch, prior = NULL,
                               verbose = 1, file = NULL, ...)
{
    ## Basics
    file.given <- !is.null(file)
    file.xsts <- file.exists(file)

    ## If file was provided and exists, load it.
    if(file.given && file.xsts) {

        ## Load its objects (thus converting the GNN(s) from 'raw' to 'keras' objects)
        loadGNN(file)
        ## Note: Could check whether returned object contains a GNN and
        ##       whether that's trained (and if not we could even train it),
        ##       but since 'file' most likely was saved by this function, there
        ##       is no need and it's also somewhat less fail-save to do this.

    } else {

        if(inherits(x, "gnn_GNN")) { # single GNN
            x <- fitGNN(x, data = data, batch.size = batch.size, n.epoch = n.epoch,
                        prior = prior, verbose = verbose, ...)
        } else { # list of GNNs
            ## Check
            is.gnn <- is.GNN(x)
            if(!any(is.gnn))
                stop("No object of class 'gnn_GNN' found in 'x'")
            whch.gnn <- which(is.gnn)
            ## Train each of the GNNs
            for(i in whch.gnn) {
                x[[i]] <- fitGNN(x[[i]], data = data, batch.size = batch.size, n.epoch = n.epoch,
                                 prior = prior, verbose = verbose, ...)
            }
        }

        ## Saving
        ## If file was provided, save the trained GNN(s) (thus converting
        ## the model component from 'keras' to 'raw')
        if(file.given) { # automatically means file does not exist, so we can save
            args <- c(x, file = file)
            do.call(saveGNN, args = args)
        }

        ## Return trained GNN(s)
        x

    }
}
