### GNN sampling generic #######################################################

rGNN <- function(GNN, ...) UseMethod("rGNN")


### GNN sampling method ########################################################

##' @title Sampling Method for Objects of Class "gnn_GNN"
##' @param GNN object of S3 class "gnn_GNN" to be sampled from with d-dimensional
##'        input layer
##' @param size sample size; chosen as nrow(prior) if prior is a matrix
##' @param prior NULL, (n, d)-matrix (or d-vector) of samples or a function of 'n'
##' @param copula object of S4 class 'Copula' (d-dimensional); has to the
##'        independence, Clayton, normal or t copula if method = "sobol"
##' @param margins d-list of marginal quantile functions or a single one which is
##'        then repeated d times.
##' @param method character string indicating the method to be used for sampling
##' @param ... additional arguments passed to 'method'
##' @return Sample from the GNN (feedforwarded input sample)
##' @author Marius Hofert
##' @note rGNN.numeric <- function(n, GNN, prior, ...) would have been another
##'       option but then 'n' is required even if prior is a sample, which is
##'       weird. And omitting 'n' then leads to error 'no applicable method for 'rGNN'
##'       applied to an object of class "name"'. The current version acts more
##'       like sample()
rGNN.gnn_GNN <- function(GNN, size, prior = NULL, copula = indepCopula(dim(GNN)[1]),
                         margins = qnorm, method = c("pseudo", "sobol"), ...)
{
    stopifnot(inherits(GNN, "gnn_GNN"))
    d <- dim(GNN)[1]
    if(is.null(prior)) { # prior = NULL

        if(!inherits(copula, "Copula"))
            stop("'copula' must be of class 'Copula'")
        if(is.function(margins))
            margins <- rep(list(margins), d)
        if(!all(sapply(margins, is.function)))
            stop("'margins' must be a quantile function or vector of dimension dim(GNN)[1] of such.")
        ## Generate copula sample
        method <- match.arg(method)
        U <- switch(method,
                    "pseudo" = {
                        rCopula(size, copula)
                    },
                    "sobol" = {
                        if(missing(randomize))
                            randomize <- "digital.shift"
                        if(missing(seed))
                            seed <- 271
                        U. <- sobol(size, d = d, randomize = randomize, seed = seed, ...)
                        if(!inherits(copula, "indepCopula") && # those having an inverse Rosenblatt transform
                           !inherits(copula, "claytonCopula") &&
                           !inherits(copula, "normalCopula") &&
                           !inherits(copula, "tCopula"))
                            stop("For method = \"sobol\", 'copula' must currently be an independence, Clayton, normal or t copula.")
                        cCopula(U., copula = copula, inverse = TRUE)
                    },
                    stop("Wrong 'method'."))
        ## Map to the given margins
        prior <- sapply(1:d, function(j) margins[[j]](U[,j]))
        if(!is.matrix(prior)) # for size = 1
            prior <- rbind(prior, deparse.level = 0L)

    } else { # 'prior' was provided

        if(is.numeric(prior)) {
            ## Nothing to do here as the below code works in this case
        } else if(is.function(prior)) { # 'prior' is a function
            prior <- prior(n)
        } else stop("'prior' must be numeric (vector or matrix) or a sampling function")
        ## Sanity checks
        if(!is.matrix(prior))
            prior <- rbind(prior, deparse.level = 0L)
        if(ncol(prior) != d)
            stop("Number of columns of 'prior' matrix does not match dim(GNN)[1]")

    }

    ## Feedforward
    feedforward(GNN, data = prior)
}
