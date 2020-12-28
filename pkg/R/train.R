### Training GNNs ##############################################################

##' @title Training GNNs
##' @param x GNN object as returned by constructors
##' @param data (n,d)-matrix containing n d-dimensional observations forming the
##'        training data
##' @param batch.size number of samples per stochastic gradient step
##' @param n.epoch number of epochs (one epoch equals one pass through the complete
##'        training dataset while updating the GNN's parameters)
##' @param verbose see ?keras:::fit.keras.engine.training.Model
##' @param ... additional arguments passed to the underlying fit()
##'        see ?keras:::fit.keras.engine.training.Model
##' @return trained GNN
##' @author Marius Hofert
train <- function(x, data, batch.size, n.epoch, verbose = 3, ...)
{
    ## Define variables and do checks
    if(!is.matrix(data))
        stop("'data' needs to be an (n, d)-matrix containing n d-dimensional training observations.")
    dim.train <- dim(data) # training data dimensions
    stopifnot(1 <= batch.size, batch.size <= dim.train[1], n.epoch >= 1)
    nms <- names(x)
    if(!("type" %in% nms && "dim" %in% nms))
        stop("'x' must have components 'type' and 'dim'.")
    type <- x$type
    if(type != "GMMN") # && type != "VAE")
        stop("The only GNN type currently supported is 'GMMN'.")
    dim <- x$dim
    dim.out <- switch(type, # dimension of the output layer
                      "GMMN" = {
                          dim[length(dim)]
                      },
                      ## "VAE" = {
                      ##     dim[1] # for VAEs, the dimension of input and output layers are equal
                      ## },
                      stop("Wrong 'type'"))
    if(dim.train[2] != dim.out)
        stop("The dimension of the training data does not match the dimension of the output layer of the GNN")

    ## Training (depending on model type)
    switch(type,
           "GMMN" = {
               prior <- matrix(rnorm(dim.train[1] * dim[1]), nrow = dim.train[1]) # N(0,1) prior (same dimension as input layer)
               x$model %>% fit(x = prior, y = data, # x = data (here: prior, could also be user input) passed through NN as input; y = target/training data (e.g., copula data)
                                 batch_size = batch.size, epochs = n.epoch, verbose = verbose, ...) # training
           },
           ## "VAE" = {
           ##     x$model %>% fit(x = data, y = data, # both input and output to the NN are the target/training data
           ##                       batch_size = batch.size, epochs = n.epoch, verbose = verbose, ...)
           ## },
           stop("Wrong 'type'"))

    ## Update information
    x[["n.train"]] <- dim.train[1]
    x[["batch.size"]] <- batch.size
    x[["n.epoch"]] <- n.epoch

    ## Return
    x # GNN object with trained model and additional information
}

##' @title Training or Loading a Trained GNN
##' @param x see ?train
##' @param data see ?train
##' @param batch.size see ?train
##' @param n.epoch see ?train
##' @param file character string (with or without ending .rda) specifying the file
##'        to save the trained GNN object to (with component 'model' serialized)
##' @param name name under which the trained GNN object is saved in 'file'
##' @param package name of the package from which to read the trained GNN; if NULL
##'        (the default) the current working directory is used.
##' @param ... additional arguments passed to the underlying train()
##' @return trained or loaded GNN object
##' @author Marius Hofert
train_once <- function(x, data, batch.size, n.epoch,
                       file, name = rm_ext(basename(file)), package = NULL, ...)
{
    if(exists_rda(file, names = name, package = package)) { # check existence of 'name' in 'file'
        read.x <- read_rda(file, names = name, package = package) # GNN object with serialized component 'model'
        if(read.x[["type"]] != x[["type"]])
            stop("The 'type' of the read GNN and that of 'x' do not coincide")
        as.keras(read.x) # return whole GNN object (with unserialized model (components))
    } else { # if 'file' does not exist or 'name' does not exist in 'file'
        ## Train and update training slots
        trained.x <- train(x, data = data, batch.size = batch.size, n.epoch = n.epoch, ...) # trained GNN
        ## Convert necessary slots to storable objects
        trained.x. <- as.raw(trained.x)
        ## Save and return
        save_rda(trained.x., file = file, names = name) # save the trained model (with savable GNNs)
        trained.x # return trained GNN object (with original GNNs)
    }
}
