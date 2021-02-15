### Model super-class ##########################################################

##' @title Constructor for the Super-Class gnn_Model
##' @param name character string of the constructor of an object of class
##'        "Copula" or "gnn_GNN"
##' @param ... additional parameters passed on to the constructor
##' @return An object of class "gnn_Model"
##' @author Marius Hofert
##' @note Allows for easier comparison of copulas and neural networks
Model <- function(name, ...)
{
    model <- do.call(name, args = list(...))
    res <- if(inherits(model, "Copula")) {
        structure(list(model = model,
	               type = "Copula",
		       n.param = nParam(model),
		       method = NA_character_,
		       n.train = NA_integer_,
		       time = system.time(NULL)),
                  class = c(as.vector(class(model)), "copula", "gnn_Model"))
    } else if(inherits(model, "gnn_GNN")) {
        ## 'model' already has the right (S3) structure and inherits from 'gnn_Model'
	stopifnot("gnn_Model" %in% class(model)) # check
        model
    } else stop("Wrong 'model'")
    res
}


### Super-class random number generation #######################################

##' @title Sampling Method for Objects of Class "gnn_Model"
##' @param x object of S3 class "gnn_Model" to be sampled from
##' @param size sample size
##' @param prior NULL or a (size, d)-matrix of prior samples
##' @param pobs logical indicating whether pobs() is applied to the output
##'        before returning
##' @param ... additional arguments passed to the underlying functions
##' @return Sample from the model
##' @author Marius Hofert
rModel <- function(x, size, prior = NULL, pobs = FALSE, ...)
{
        stopifnot(n >= 0, inherits(object, "gnn_Model"))
	switch(model[["type"]],
	"FNN" = {
	     rGNN(x, size = size, prior = prior, pobs = pobs, ...)
        },
        "Copula" = {
            res <- if(is.null(prior)) {
                       rCopula(size, copula = x[["model"]], ...)
                   } else { # if 'prior' is provided, compute the inverse Rosenblatt transform
                       cCopula(prior, copula = x[["model"]], inverse = TRUE, ...)
                   }
            if(pobs) pobs(res) else res
        },
        stop("Wrong 'type'"))
}
