### GNN feedforward generic ####################################################

feedforward <- function(GNN, ...) UseMethod("feedforward")


### GNN feedfoward method ######################################################

feedforward.gnn_GNN <- function(GNN, data)
    predict(GNN[["model"]], x = data)
