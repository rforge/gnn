### GNN feedforward generic ####################################################

ffGNN <- function(x, data) UseMethod("ffGNN")


### GNN feedfoward method ######################################################

##' @title Feedforward Method for Objects of Class "gnn_GNN"
##' @param x object of S3 class "gnn_GNN" to be sampled from (input layer is
##'        d-dimensional)
##' @param data (n, d)-matrix of data to be fed forward through 'x'
##' @return the output (matrix) of the GNN 'x'
##' @author Marius Hofert
ffGNN.gnn_GNN <- function(x, data)
{
    stopifnot(inherits(x, "gnn_GNN"))
    if(!is.matrix(data))
        data <- rbind(data)
    if(ncol(data) != dim(x)[1])
        stop("ncol(data) does not match dim(x)[1]")
    predict(x[["model"]], x = data)
}
