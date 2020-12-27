### GNN feedforward generic ####################################################

ffGNN <- function(gnn, data) UseMethod("ffGNN")


### GNN feedfoward method ######################################################

##' @title Feedforward Method for Objects of Class "gnn_GNN"
##' @param gnn object of S3 class "gnn_GNN" to be sampled from (input layer is
##'        d-dimensional)
##' @param data (n, d)-matrix of data to be fed forward through 'gnn'
##' @return the output (matrix) of the GNN gnn
##' @author Marius Hofert
ffGNN.gnn_GNN <- function(gnn, data) {
    if(!is.matrix(data))
        data <- rbind(data)
    if(ncol(data) != dim(gnn)[1])
        stop("ncol(data) does not match dim(gnn)[1]")
    predict(gnn[["model"]], x = data)
}
