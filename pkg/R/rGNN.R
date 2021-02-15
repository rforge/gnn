### GNN sampling generic #######################################################

rGNN <- function(x, ...)  UseMethod("rGNN") # generic


### GNN sampling method ########################################################

##' @title Sampling Method for Objects of Class "gnn_GNN"
##' @param x object of S3 class "gnn_GNN" to be sampled from (input layer is
##'        d-dimensional)
##' @param size sample size
##' @param prior NULL (in which case N(0,1)^d is pseudo-sampled via rPrior())
##'        or a (size, d)-matrix of prior samples.
##' @param pobs logical indicating whether pobs() is applied to the output
##'        before returning
##' @param ... additional arguments passed to rPrior() if prior = NULL
##' @return Sample from the GNN 'x' (feedforwarded prior sample)
##' @author Marius Hofert
##' @note rGNN.numeric <- function(n, x, prior, ...) would have been another
##'       option but then 'n' is required even if 'prior' is a sample, which is
##'       weird. And omitting 'n' then leads to error 'no applicable method for
##'       'rGNN' applied to an object of class "name"'. The current version acts
##'       more like sample()
rGNN.gnn_GNN <- function(x, size, prior = NULL, pobs = FALSE, ...)
{
    stopifnot(inherits(x, "gnn_GNN"), is.logical(pobs))
    if(is.null(prior))
        prior <- rPrior(size, copula = indepCopula(dim(x)[1]), ...) # independent N(0,1)
    res <- ffGNN(x, data = prior)
    if(pobs) pobs(res) else res
}
