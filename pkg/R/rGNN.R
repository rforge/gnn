### GNN sampling generic #######################################################

rGNN <- function(x, ...)  UseMethod("rGNN")


### GNN sampling method ########################################################

##' @title Sampling Method for Objects of Class "gnn_GNN"
##' @param x object of S3 class "gnn_GNN" to be sampled from (input layer is
##'        d-dimensional)
##' @param size sample size; chosen as nrow(prior) if prior is a vector or matrix
##' @param prior NULL, (n, d)-matrix (or d-vector) of samples or a function of
##'        'size'
##' @param copula object of S4 class 'Copula' (d-dimensional); has to the
##'        independence, Clayton, normal or t copula if method = "sobol"
##' @param qmargin d-list of marginal quantile functions or a single one which is
##'        then repeated d times.
##' @param method character string indicating the method to be used for sampling
##' @param ... additional arguments passed to 'method'
##' @return Sample from the GNN 'x' (feedforwarded input sample)
##' @author Marius Hofert
##' @note rGNN.numeric <- function(n, x, prior, ...) would have been another
##'       option but then 'n' is required even if prior is a sample, which is
##'       weird. And omitting 'n' then leads to error 'no applicable method for
##'       'rGNN' applied to an object of class "name"'. The current version acts
##'       more like sample()
rGNN.gnn_GNN <- function(x, size, prior = NULL, copula = indepCopula(dim(x)[1]),
                         qmargin = qnorm, method = c("pseudo", "sobol"), ...)
{
    stopifnot(inherits(x, "gnn_GNN"))
    d <- dim(x)[1]
    if(is.null(prior)) { # prior = NULL

        if(!inherits(copula, "Copula"))
            stop("'copula' must be of class 'Copula'")
        if(is.function(qmargin))
            qmargin <- rep(list(qmargin), d)
        if(!all(sapply(qmargin, is.function)))
            stop("'qmargin' must be a quantile function or vector of dimension dim(x)[1] of such.")
        ## Generate copula sample
        method <- match.arg(method)
        U <- switch(method,
                    "pseudo" = {
                        rCopula(size, copula)
                    },
                    "sobol" = {
                        args <- list(...)
                        if(!hasArg("randomize"))
                            args <- c(args, randomize = "digital.shift")
                        if(!hasArg("seed"))
                            args <- c(args, seed = 271)
                        U. <- do.call(sobol, args = c(n = size, d = d, args))
                        if(!inherits(copula, "indepCopula") && # those having an inverse Rosenblatt transform
                           !inherits(copula, "claytonCopula") &&
                           !inherits(copula, "normalCopula") &&
                           !inherits(copula, "tCopula"))
                            stop("For method = \"sobol\", 'copula' must currently be an independence, Clayton, normal or t copula.")
                        cCopula(U., copula = copula, inverse = TRUE)
                    },
                    stop("Wrong 'method'."))
        ## Map to the given margins
        prior <- sapply(1:d, function(j) qmargin[[j]](U[,j]))
        if(!is.matrix(prior)) # for size = 1
            prior <- rbind(prior, deparse.level = 0L)

    } else { # 'prior' was provided

        if(is.numeric(prior)) {
            ## Nothing to do here as the below code works in this case
        } else if(is.function(prior)) { # 'prior' is a function
            prior <- prior(size)
        } else stop("'prior' must be numeric (vector or matrix) or a sampling function")
        ## Sanity checks
        if(!is.matrix(prior))
            prior <- rbind(prior, deparse.level = 0L)
        if(ncol(prior) != d)
            stop("Number of columns of 'prior' matrix does not match dim(x)[1]")

    }

    ## Feedforward
    ffGNN(x, data = prior)
}
