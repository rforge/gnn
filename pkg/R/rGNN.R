### GNN sampling generic #######################################################

rGNN <- function(x, ...)  UseMethod("rGNN")


### GNN sampling method ########################################################

##' @title Sampling Method for Objects of Class "gnn_GNN"
##' @param x object of S3 class "gnn_GNN" to be sampled from (input layer is
##'        d-dimensional)
##' @param size sample size
##' @param prior NULL (in which case N(0,1)^d is pseudo-sampled via rPrior())
##'        or a (size, d)-matrix of prior samples.
##' @param ... additional arguments passed to rPrior() if prior = NULL
##' @return Sample from the GNN 'x' (feedforwarded input sample)
##' @author Marius Hofert
##' @note rGNN.numeric <- function(n, x, prior, ...) would have been another
##'       option but then 'n' is required even if prior is a sample, which is
##'       weird. And omitting 'n' then leads to error 'no applicable method for
##'       'rGNN' applied to an object of class "name"'. The current version acts
##'       more like sample()
rGNN.gnn_GNN <- function(x, size, prior = NULL, ...)
{
    stopifnot(inherits(x, "gnn_GNN"), size >= 1)
    if(is.null(prior))
        prior <- rPrior(size, copula = indepCopula(dim(x)[1]), ...) # independent N(0,1)
    ffGNN(x, data = prior)
}
