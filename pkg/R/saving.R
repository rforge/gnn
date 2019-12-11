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
                                                   custom_objects = c(loss = loss, loss_fn = loss)) # used to be loss = loss (and loss_fn = loss) when run interactively, but suddenly stopped to work (2019-10-06).
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
