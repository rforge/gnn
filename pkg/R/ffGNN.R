### GNN feedforward generic ####################################################

ffGNN <- function(GNN, data) UseMethod("ffGNN")


### GNN feedfoward method ######################################################

##' @title Feedforward Method for Objects of Class "gnn_GNN"
##' @param GNN object of S3 class "gnn_GNN" to be sampled from (input layer is
##'        d-dimensional)
##' @param data (n, d)-matrix of data to be fed forward through 'GNN'
##' @return the output (matrix) of the GNN
##' @author Marius Hofert
ffGNN.gnn_GNN <- function(GNN, data) {
    if(!is.matrix(data))
        data <- rbind(data)
    if(ncol(data) != dim(GNN)[1])
        stop("ncol(data) does not match dim(GNN)[1]")
    predict(GNN[["model"]], x = data)
}
