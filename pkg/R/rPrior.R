### Sampling from prior distribution ###########################################

##' @title Sampling from Prior
##' @param n sample size
##' @param copula object of S4 class 'Copula' (d-dimensional); if method = "sobol",
##'        this has to be the independence, Clayton, normal or t copula
##' @param qmargins d-list of marginal quantile functions or a single one which is
##'        then repeated d times.
##' @param method character string indicating the method to be used for sampling
##' @param ... additional arguments passed to 'method'
##' @return (n, d)-matrix
##' @author Marius Hofert
##' @note 'copula' has to be provided to specify the dimension of the sample
rPrior <- function(n, copula, qmargins = qnorm, method = c("pseudo", "sobol"), ...)
{
    ## Checks
    stopifnot(n >= 1)
    if(!inherits(copula, "Copula"))
        stop("'copula' must be of class 'Copula'")
    d <- dim(copula)
    if(is.character(qmargins))
        stop("'qmargins' must be a quantile function (not string) or vector of such")
    if(is.function(qmargins))
        qmargins <- rep(list(qmargins), d)
    if(length(qmargins) != d)
        stop("length(qmargins) != dim(copula)")
    if(!all(sapply(qmargins, is.function)))
        stop("'qmargins' must be a quantile function or vector of dimension dim(x)[1] of such.")

    ## Generate copula sample
    method <- match.arg(method)
    U <- switch(method,
                "pseudo" = {
                    rCopula(n, copula = copula)
                },
                "sobol" = {
                    args <- list(...)
                    if(!hasArg("randomize"))
                        args <- c(args, randomize = "digital.shift")
                    U. <- do.call(sobol, args = c(n = n, d = d, args))
                    if(!inherits(copula, "indepCopula") && # those having an inverse Rosenblatt transform
                       !inherits(copula, "claytonCopula") &&
                       !inherits(copula, "normalCopula") &&
                       !inherits(copula, "tCopula"))
                        stop("For method = \"sobol\", 'copula' must currently be an independence, Clayton, normal or t copula.")
                    cCopula(U., copula = copula, inverse = TRUE)
                },
                stop("Wrong 'method'."))

    ## Map U to the given margins
    prior <- sapply(1:d, function(j) qmargins[[j]](U[,j]))
    if(!is.matrix(prior)) # for n = 1
        prior <- rbind(prior, deparse.level = 0L)
    prior
}

