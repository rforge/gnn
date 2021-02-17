### GNN training R6 class for progress #########################################

## See keras/R/callbacks.R, https://keras.rstudio.com/reference/KerasCallback.html
## and https://cran.r-project.org/web/packages/keras/vignettes/training_callbacks.html
progress <-
    R6Class("gnn_progress",
            inherit = KerasCallback,
            public = list(
                n.epoch = integer(1), # argument of new()
                verbose = integer(1), # argument of new()
                pb = NULL, # provided to avoid "cannot add bindings to a locked environment"
                initialize = function(n.epoch, verbose, pb = NULL) { # checks, executed by new()
                    ## Note: - 'pb' needs to be provided to avoid "object 'pb' not found"
                    ##       - give 'pb' a default argument to not require it to be provided in new()
                    stopifnot(n.epoch >= 1,
                              verbose %% 1 == 0, 0 <= verbose, verbose <= 3)
                    self$n.epoch <- n.epoch
                    self$verbose <- verbose
                },
                on_train_begin = function(logs) { # 'logs' is required
                    if(self$verbose == 1)
                        self$pb <- txtProgressBar(max = self$n.epoch)
                },
                on_epoch_end = function(epoch, logs = list()) { # 'epoch' and 'logs' are required
                    epo <- epoch + 1 # epoch starts from 0
                    switch(self$verbose + 1, # in {1,2,3,4}
                    { # verbose = 0
                                        # silent => do nothing
                    },
                    { # verbose = 1
                        setTxtProgressBar(self$pb, epo)
                    },
                    { # verbose = 2
                        ndigits <- floor(log10(self$n.epoch)) + 1
                        fmt.strng <- paste0("Epoch %",ndigits,"d/%d finished with loss %f\n")
                        mult <- function(n) # output is written at multiple of 'mult'
                            ifelse(n <= 100, ceiling(n/10), ceiling(sqrt(n)))
                        div <- mult(self$n.epoch)
                        if(epo %% div == 0)
                            cat(sprintf(fmt.strng, epo, self$n.epoch, logs[["loss"]]))
                    },
                    { # verbose = 3
                        ndigits <- floor(log10(self$n.epoch)) + 1
                        fmt.strng <- paste0("Epoch %",ndigits,"d/%d finished with loss %f\n")
                        cat(sprintf(fmt.strng, epo, self$n.epoch, logs[["loss"]]))
                    },
                    stop("Wrong 'verbose'"))
                    },
                    on_train_end = function(logs) { # 'logs' is required
                        if(self$verbose == 1) close(self$pb)
                    }
    ))


### GNN training generics ######################################################

## Generic for main training routine
fitGNN <- function(x, data, ...) UseMethod("fitGNN")

## Generic for training once
fitGNNonce <- function(x, data, ...) UseMethod("fitGNNonce")

## Generic for checking being trained
is.trained <- function(x) UseMethod("is.trained")


### GNN training methods #######################################################

##' @title Training GNNs
##' @param x object of class gnn_GNN as returned by constructor(s)
##' @param data (n, d)-matrix containing n d-dimensional observations forming the
##'        training data
##' @param batch.size number of samples per stochastic gradient step
##' @param n.epoch number of epochs (one epoch equals one pass through the complete
##'        training dataset while updating the GNN's parameters)
##' @param prior (n, d)-matrix of prior samples (if NULL, iid N(0,1) data)
##' @param max.n.prior maximum number of prior samples stored in x
##' @param verbose choices are:
##'        0 = silent
##'        1 = progress bar
##'        2 = output of max. 10 epochs and their losses
##'        3 = output of each epoch and its loss
##'        Note that we internally use fit()'s (= keras:::fit.keras.engine.training.Model)
##'        verbose = 0 to suppress its (non-ideal) output.
##' @param ... additional arguments passed to the underlying fit.keras.engine.training.Model;
##'        see keras:::fit.keras.engine.training.Model or ?fit.keras.engine.training.Model
##' @return trained GNN
##' @author Marius Hofert
fitGNN.gnn_GNN <- function(x, data, batch.size = nrow(data), n.epoch = 100, prior = NULL,
                           max.n.prior = 5000, verbose = 2, ...)
{
    ## Checks
    if(!is.matrix(data))
        stop("'data' needs to be an (n, d)-matrix containing n d-dimensional training observations.")
    dim.train <- dim(data) # training data dimensions
    stopifnot(inherits(x, "gnn_GNN"),
              1 <= batch.size, batch.size <= dim.train[1], n.epoch >= 1,
              verbose %% 1 == 0, 0 <= verbose, verbose <= 3)
    dim.out <- tail(dim(x), n = 1) # output dimension
    ## Note: for VAEs, dim.out = dim(x)[1] as input and output layer have the same dim
    if(dim.train[2] != dim.out)
        stop("The dimension of the training data does not match the dimension of the output layer of the GNN")

    ## Train and possibly save
    type <- x[["type"]]
    switch(type,
           "FNN" = {
               if(is.null(prior)) {
                   prior <- rPrior(nrow(data), copula = indepCopula(ncol(data))) # independent N(0,1)
               } else {
                   if(!all(dim(data) == dim(prior)))
                       stop("dim(data) != dim(prior)")
               }
               ## Note:
               ## - x = data to be passed through NN as input
               ##   y = target/training data to compare against
               ## - fit() modifies x[["model"]] in place
               has.callbacks <- hasArg("callbacks")
               callbacks <-  if(!has.callbacks) list() else list(...)$callbacks # take callbacks...
               callbacks <- c(progress$new(n.epoch = n.epoch,
                                           verbose = verbose), callbacks) # ... and concatenate progress output object
               ## Probably not worth checking whether 'callbacks' already contains a
               ## progress-type object
               args <- list(object = x[["model"]], x = prior, y = data, # see ?fit.keras.engine.training.Model
                            batch_size = batch.size, epochs = n.epoch, verbose = 0, # silent, but...
                            callbacks = callbacks) # ... progress determined through callback
               dots <- list(...)
               if(has.callbacks) dots$callbacks <- NULL # remove callbacks from '...'
               tm <- system.time(history <- do.call(fit, args = c(args, dots)))

               ## Update slots of 'x'
               x[["n.train"]] <- dim.train[1]
               x[["batch.size"]] <- batch.size
               x[["n.epoch"]] <- n.epoch
               x[["loss"]] <- history[["metrics"]][["loss"]] # (only interesting component in there)
               x[["time"]] <- tm
               x[["prior"]] <- prior[seq_len(min(nrow(prior), max.n.prior)),] # grab out at most max.n.prior samples
           },
           ## "VAE" = {
           ##     ## Note:
           ##     ## - Not updated (callbacks, time measurement etc.)
           ##     ## - Both input and output to the NN are the target/training data
           ##     fit(x[["model"]], x = data, y = data,
           ##         batch_size = batch.size, epochs = n.epoch, verbose = 0, ...)
           ## },
           stop("Wrong 'type'"))

    ## Return trained GNN
    x
}

##' @title Training GNNs with Saving and Restoring
##' @param x object of class gnn_GNN. The GNN is trained (even if already trained)
##'        and then saved, unless 'file' is provided and exists in which case 'file'
##'        is loaded and returned.
##' @param data see fitGNN.gnn_GNN()
##' @param batch.size see fitGNN.gnn_GNN()
##' @param n.epoch see fitGNN.gnn_GNN()
##' @param prior see fitGNN.gnn_GNN()
##' @param verbose see fitGNN.gnn_GNN()
##' @param file NULL or a file name in which case the trained GNN is saved in the
##'        provided file. If called again and the file exists, no training is done
##'        but the trained object loaded from the file.
##' @param name character string giving the name under which the fitted 'x' is saved
##' @param ... see fitGNN.gnn_GNN()
##' @return trained GNN
##' @author Marius Hofert
fitGNNonce.gnn_GNN <- function(x, data, batch.size = nrow(data), n.epoch = 100,
                               prior = NULL, verbose = 2, file = NULL, name = NULL, ...)
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

        ## Training
        x <- fitGNN(x, data = data, batch.size = batch.size, n.epoch = n.epoch,
                    prior = prior, verbose = verbose, ...)

        ## Saving
        ## If file was provided, save the trained GNN (thus converting
        ## the model component from 'keras' to 'raw')
        if(file.given)
            saveGNN(x, file = file, name = name)

        ## Return trained GNN(s)
        x

    }
}

##' @title Check for Being an Objects of Class "gnn_GNN" which is Trained
##' @param x R object
##' @return logical indicating whether 'x' is of class "gnn_GNN" and trained
##' @author Marius Hofert
##' @note similar to is.GNN.gnn_GNN()
is.trained.gnn_GNN <- function(x)
{
    if(inherits(x, "gnn_GNN")) {
        !is.na(x[["n.train"]])
    } else stop("'x' is not an object of class \"gnn_GNN\"")
}

##' @title Check for Being a List of Objects of Class "gnn_GNN" which are Trained
##' @param x R object
##' @return logical indicating whether 'x' is a list of objects of class "gnn_GNN"
##'         which are trained
##' @author Marius Hofert
##' @note similar to is.GNN.list()
is.trained.list <- function(x)
{
    if(inherits(x, "list")) {
        sapply(x, function(x.) {
            inherits(x., "gnn_GNN") && !is.na(x.[["n.train"]])
        })
    } else stop("'x' is not a list")
}
