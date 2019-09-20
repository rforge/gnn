### Training GNNs ##############################################################

##' @title Training GNNs
##' @param gnn GNN object as returned by GMMN_model() or VAE_model()
##' @param data (n,d)-matrix containing n d-dimensional observations forming the
##'        training data
##' @param batch.size number of samples per stochastic gradient step
##' @param nepoch number of epochs (one epoch equals one pass through the complete
##'        training dataset while updating the GNN's parameters)
##' @return trained GNN
##' @author Marius Hofert
train <- function(gnn, data, batch.size, nepoch)
{
    ## Define variables and do checks
    if(!is.matrix(data))
        stop("'data' needs to be an (n, d)-matrix containing n d-dimensional training observations.")
    dim.train <- dim(data) # training data dimensions
    stopifnot(1 <= batch.size, batch.size <= dim.train[1], nepoch >= 1)
    nms <- names(gnn)
    if(!("dim" %in% nms && "type" %in% nms))
        stop("'gnn' must have components 'dim' and 'type'.")
    type <- gnn$type
    if(type != "GMMN" && type != "VAE")
        stop("The only GNN types currently supported are 'GMMN' and 'VAE'.")
    dim <- gnn$dim
    dim.out <- switch(type, # dimension of the output layer
                      "GMMN" = {
                          dim[length(dim)]
                      },
                      "VAE" = {
                          dim[1] # for VAEs, the dimension of input and output layers are equal
                      },
                      stop("Wrong 'type'"))
    if(dim.train[2] != dim.out)
        stop("The dimension of the training data does not match the dimension of the output layer of the GNN")

    ## Training (depending on model type)
    switch(type,
           "GMMN" = {
               prior <- matrix(rnorm(dim.train[1] * dim[1]), nrow = dim.train[1]) # N(0,1) prior (same dimension as input layer)
               gnn$model %>% fit(x = prior, y = data, # x = data (here: prior, could also be user input) passed through NN as input; y = target/training data (e.g., copula data)
                                 batch_size = batch.size, epochs = nepoch) # training
           },
           "VAE" = {
               gnn$model %>% fit(x = data, y = data, # both input and output to the NN are the target/training data
                                 batch_size = batch.size, epochs = nepoch)
           },
           stop("Wrong 'type'"))

    ## Update information
    gnn[["dim.train"]] <- dim.train
    gnn[["batch.size"]] <- batch.size
    gnn[["nepoch"]] <- nepoch

    ## Return
    gnn # GNN object with trained model and additional information
}

##' @title Training or Loading a Trained GNN
##' @param gnn see ?train
##' @param data see ?train
##' @param batch.size see ?train
##' @param nepoch see ?train
##' @param file character string (with or without ending .rda) specifying the file
##'        to save the trained GNN object to (with component 'model' serialized)
##' @param name name under which the trained GNN object is saved in 'file'
##' @param package name of the package from which to read the trained GNN; if NULL
##'        (the default) the current working directory is used.
##' @return trained or loaded GNN object
##' @author Marius Hofert
train_once <- function(gnn, data, batch.size, nepoch,
                       file, name = rm_ext(basename(file)), package = NULL)
{
    if(exists_rda(file, names = name, package = package)) { # check existence of 'name' in 'file'
        read.gnn <- read_rda(file, names = name, package = package) # GNN object with serialized component 'model'
        if(read.gnn[["type"]] != gnn[["type"]])
            stop("The 'type' of the read GNN and that of 'gnn' do not coincide")
        to_callable(read.gnn) # return whole GNN object (with unserialized model (components))
    } else { # if 'file' does not exist or 'name' does not exist in 'file'
        ## Train and update training slots
        trained.gnn <- train(gnn, data = data, batch.size = batch.size, nepoch = nepoch) # trained GNN
        ## Convert necessary slots to storable objects
        trained.gnn. <- to_savable(trained.gnn)
        ## Save and return
        save_rda(trained.gnn., file = file, names = name) # save the trained model (with savable GNNs)
        trained.gnn # return trained GNN object (with original GNNs)
    }
}
