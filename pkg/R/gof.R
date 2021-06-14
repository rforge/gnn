### Constructor for a goodness-of-fit test #####################################

##' @title Constructor of a Two-Sample Goodness-of-Fit Test
##' @param x (n, d)-matrix of samples (or a vector which is interpreted as a
##'        one-column matrix)
##' @param y (m, d)-matrix of samples (or a vector which is interpreted as a
##'        one-column matrix)
##' @param B number of bootstrap replications for determining the p-value
##' @param method statistic to be used for the test (a string that
##'        specifies the maximum mean discrepancy (MMD) or the
##'        Cramer--von Mises (CvM) statistic)
##' @param ... additional arguments passed to the underlying test statistic
##' @return object of class "htest"
##' @author Marius Hofert
##' @note To find print.htest, do:
##'       methods("print")
##'       getAnywhere("print.htest")
gof2sample <- function(x, y, B = 1000, method = c("MMD", "CvM"), progress = TRUE, ...)
{
    ## Basics
    if(!is.matrix(x)) x <- cbind(x)
    if(!is.matrix(y)) y <- cbind(y)
    dim.x <- dim(x)
    dim.y <- dim(y)
    stopifnot(dim.x[2] == dim.y[2], dim.x[1] >= 1, dim.y[1] >= 1, B >= 1, is.logical(progress))
    method <- match.arg(method)
    if(progress)
        div <- ifelse(B <= 100, ceiling(B/10), ceiling(sqrt(B)))

    ## Compute the test statistic and corresponding string
    switch(method,
           "MMD" = {
               stat <- MMD(x, y = y, ...)
               strng <- "Maximum Mean Discrepancy (MMD)"
           },
           "CvM" = {
               stat <- CvM(x, y = y)
               strng <- "Cramer-von Mises (CvM)"
           },
           stop("Wrong 'method'"))

    ## Bootstrap
    xy <- rbind(x, y) # concatenate for this bootstrap
    n <- nrow(xy)
    stat. <- sapply(seq_len(B), function(b) {
        ## Create bootstrap samples
        xy. <- xy[sample(1:n, size = n, replace = TRUE),] # create (large) bootstrap sample
        x. <- xy.[1:dim.x[1],] # bootstrap sample
        y. <- xy.[(dim.x[1]+1):n,] # bootstrap sample
        ## Compute test statistic
        res. <- switch(method,
                       "MMD" = {
                           MMD(x., y = y., ...)
                       },
                       "CvM" = {
                           CvM(x., y = y.)
                       },
                       stop("Wrong 'method'"))
        ## Progress and return
        if(progress && (b %% div) == 0)
            cat(sprintf("%3d%% done\n", ceiling(b/B * 100)))
        res.
    })

    ## Compute result
    res <- list(
        "p.value" = (sum(stat >= stat.) + 0.5) / (B + 1),
        "statistic" = c("statistic" = stat), # name again here as otherwise not printed by print.htest
        "method" = strng,
        "B" = B, # not known to standard "htest" object (so not printed by default)
        "data.name" = paste0(c(deparse(substitute(x)),", ",deparse(substitute(y)))), # single string
        "data.dim" = c("x" = list(dim.x), "y" = list(dim.y))) # not known to standard "htest" object (so not printed by default)

    ## Return
    structure(res, class = c("htest2", "htest")) # give it also "htest2" so that we can have a better print method
}


### Methods ####################################################################

##' @title Print Method for Objects of Class "htest2"
##' @param x object of class "htest2"
##' @param digits see print.htest()
##' @param prefix see print.htest()
##' @param ... not used; for compatibility with 'print' generic
##' @return x (invisibly)
##' @author Marius Hofert
print.htest2 <- function(x, digits = getOption("digits"), prefix = "\t", ...)
{
    ## Version 1 (re-using print method of htest):
    ## x.htest <- structure(x, class = "htest") # remove "htest2"
    ## out.htest <- capture.output(print(x.htest)) # print as "htest" object and capture output
    ## if(out.htest[length(out.htest)] == "")
    ##     out.htest <- head(out.htest, n = -1) # remove last empty line
    ## cat("\n")
    ## cat(out.htest, sep = "\n") # print htest output
    ## cat(paste0("B = ",x$B,"\n"))
    ## cat(paste0("dimensions: (",x$data.dim$x[1],", ",x$data.dim$x[2],"), ",
    ##        "(",x$data.dim$y[1],", ",x$data.dim$y[2],")\n"))

    ## Version 2 (new print method; see also getAnywhere("print.htest")):
    cat("\n")
    cat(strwrap(x$method, prefix = prefix), sep = "\n")
    cat("\n")
    cat(paste0("statistic = ",format(x$statistic, digits = max(1L, digits - 2L)),
               ", p-value = ",format.pval(x$p.value, digits = max(1L, digits - 3L)),"\n"))
    cat(paste0("B = ",x$B,"\n"))
    cat("data: ", x$data.name, "\n", sep = "")
    cat(paste0("dimensions: (",x$data.dim$x[1],", ",x$data.dim$x[2],"), ",
               "(",x$data.dim$y[1],", ",x$data.dim$y[2],")\n"))
    invisible(x)
}
