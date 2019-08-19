### Training GNNs ##############################################################

##' @title Training GNNs
##' @param dim see ?GMMN_model or ?VAE_model
##' @param data (n,d)-matrix containing n d-dimensional observations forming the
##'        training data
##' @param batch.size number of samples per stochastic gradient step
##' @param nepoch number of epochs (one epoch equals one pass through the complete
##'        training dataset while updating the GNN's parameters)
##' @param method the type of GNN to train
##' @param ... additional arguments passed to the underlying GMMN_model()
##' @return trained GNN
##' @author Marius Hofert
train <- function(dim, data, batch.size, nepoch, method = c("GMMN", "VAE"), ...)
{
    method <- match.arg(method)
    if(!is.matrix(data))
        stop("'data' needs to be an (n, d)-matrix containing n d-dimensional training observations.")
    dim.train <- dim(data) # training data dimensions
    if(dim.train[2] != dim[1])
        stop("The dimension ncol(data) of the training data does not match the dimension dim[1] of the GNN")
    prior <- matrix(rnorm(dim.train[1] * dim[1]), nrow = dim.train[1]) # N(0,1) prior (same dimension as input layer)
    switch(method, # model (and checks) to train
           "GMMN" = {
               GNN <- GMMN_model(dim, ...)
               GNN %>% fit(x = prior, y = data, batch_size = batch.size, epochs = nepoch) # training
           },
           "VAE" = {
               GNN <- VAE_model(dim, ...)
               GNN$model %>% fit(x = data, y = data, # ... for how we defined the loss function
                                 batch_size = batch.size, epochs = nepoch)
           },
           stop("Wrong 'method'"))
    GNN
}

##' @title Training or loading a trained GNN
##' @param dim see ?GMMN_model or ?VAE_model
##' @param data see ?train
##' @param batch.size see ?train
##' @param nepoch see ?train
##' @param method see ?train
##' @param file character string (with or without ending .rda) specifying the file
##'        to save the trained GNN to
##' @param name name under which the trained GNN is saved in 'file'
##' @param package package name from which to load the object or NULL (the default)
##'        in which case the current working directory is searched.
##' @param ... additional arguments passed to the underlying train()
##' @return trained or loaded GNN
##' @author Marius Hofert
train_once <- function(dim, data, batch.size, nepoch, method = c("GMMN", "VAE"),
                       file, name = rm_ext(basename(file)),
                       package = NULL, ...)
{
    if(exists_rda(file, objnames = name, package = package)) { # check existence of 'name' in 'file' (in package 'package' or current working directory if package = NULL)
        unserialize_model(read_rda(name, file = file, package = package),
                          custom_objects = c(loss = loss)) # get and return
    } else { # does not exist
        GNN <- train(dim, data = data, batch.size = batch.size, nepoch = nepoch,
                     method = method, ...) # train
        save_rda(serialize_model(GNN), file = file, names = name) # save (by side-effect)
        GNN
    }
}
