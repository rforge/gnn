### Data transformations for training/sampling #################################

##' @title Marginal Range Transformation
##' @param x (n, d)-matrix of data (typically before training or after sampling)
##' @param lower value or d-vector typically containing the smallest value of
##'        each column of x
##' @param upper value or d-vector typically containing the largest value of
##'        each column of x
##' @param inverse logical indicating whether a range transform (the default)
##'        or its inverse is applied to each component sample of x
##' @return (n, d)-matrix of marginally transformed data
##' @author Marius Hofert
range_trafo <- function(x, lower, upper, inverse = FALSE)
{
    ## Basics
    if(!is.matrix(x))
        x <- cbind(x) # one column vector
    d <- ncol(x)
    if(length(lower) == 1) lower <- rep(lower, d)
    if(length(upper) == 1) upper <- rep(upper, d)

    ## Marginally apply the transformation
    if(!inverse) {
        sapply(1:d, function(j) (x[,j] - lower[j]) / (upper[j] - lower[j]))
    } else {
        sapply(1:d, function(j) lower[j] + (upper[j] - lower[j]) * x[,j])
    }
}

##' @title Marginal Linearly Transformed Logistic Function
##' @param x (n, d)-matrix of data (typically before training or after sampling)
##' @param mean value or d-vector of marginal sample means
##' @param sd value or d-vector of marginal sample standard deviations
##' @param slope value or d-vector of slopes of the linear transformations
##'        applied after applying plogis() (if inverse: before applying qlogis())
##' @param intercept value or d-vector of intercepts of the linear
##'        transformations applied after applying plogis() (if inverse: before
##'        applying qlogis())
##' @param inverse logical indicating whether a linear transformation of plogit()
##'        (the default) or qlogit() of a linear transformation of the data is
##'        applied to each component sample of x
##' @return (n, d)-matrix of marginally transformed data (linear transformations
##'         of plogis() if !inverse and qlogis() of linear transformations of the
##'         data if inverse)
##' @author Marius Hofert
##' @note 1) On each margin, the (log-)logistic distribution/quantile function
##'          (= inverse logit/logit function) is used with location and scale chosen
##'          to match the provided (sample) mean and standard deviation.
##'       2) We need to provide 'mean' and 'sd' as arguments (instead of computing
##'          them) for the case inverse = TRUE (the sample mean or sd of the original
##'          x is unknown then, so we can't transform back)
##'       3) We can't have a log = FALSE argument here as it is unclear how 'mean'
##'          and 'sd' change. For the same reason as 2), we can't use the available
##'          data to recompute 'mean' and 'sd' (as inverse = TRUE would not work then).
##'          The easiest is to first manually log the data, then compute 'mean' and 'sd'
##'          and call logis_trafo().
logis_trafo <- function(x, mean = 0, sd = 1, slope = 1, intercept = 0,
                        inverse = FALSE)
{
    ## Basics
    if(!is.matrix(x))
        x <- cbind(x) # one column vector
    d <- ncol(x) # dimension of x
    if(length(mean) == 1) mean <- rep(mean, d)
    if(length(sd) == 1) sd <- rep(sd, d)
    if(length(intercept) == 1) intercept <- rep(intercept, d)
    if(length(slope) == 1) slope <- rep(slope, d)
    stopifnot(d >= 1,
              length(mean) == d, length(sd) == d, sd > 0,
              length(intercept) == d, length(slope) == d, slope > 0,
              is.logical(inverse))

    ## Marginally apply the transformation
    if(!inverse) {
        sapply(1:d, function(j)
            intercept[j] + slope[j] * plogis(x[,j], location = mean[j],
                                             scale = sqrt(3*sd[j]^2/pi^2)))
    } else { # inverse
        sapply(1:d, function(j)
            qlogis((x[,j] - intercept[j]) / slope[j], location = mean[j],
                   scale = sqrt(3*sd[j]^2/pi^2)))
    }
}


### Principal component analysis ###############################################

##' @title Principal Component Transformtion
##' @param x (n, d)-matrix of data (typically before training or after sampling).
##'        If inverse, then an (n, k)-matrix with 1 <= k <= d.
##' @param mu d-vector for the transformation Y = Gamma^T (X - mu)
##' @param Gamma (d, k)-matrix with k >= ncol(x) whose columns contain k orthonormal
##'        eigenvectors of a covariance matrix sorted in decreasing order of their
##'        eigenvalues. If a matrix with k > ncol(x) is provided, only the first
##'        k-many are considered.
##' @param inverse logical indicating whether the inverse transformation is applied
##'        based on provided 'mu' and 'Gamma'.
##' @param ... additional arguments passed to the underlying prcomp() if inverse.
##' @return if inverse: list with components:
##'         "PCs": principal components or 'scores' (Y = Gamma^T (X - mu))
##'         "cumvar": cumulative variances
##'         "sd": standard deviations of each Y;
##'         "lambda": eigenvalues of cov(x) sorted in decreasing order;
##'         "mu": computed centers;
##'         "Gamma": computed matrix of sorted orthonormal eigenvectors;
##'         otherwise: transformed (n, d)-matrix of data
##' @author Marius Hofert
##' @note See also MFE (2015, Section 6.4.5)
PCA_trafo <- function(x, mu, Gamma, inverse = FALSE, ...)
{
    stopifnot(is.matrix(x), ncol(x) >= 1, is.logical(inverse))
    if(!inverse) { # unused: mu, Gamma
        ## PCA
        PCA <- prcomp(x, ...)

        ## Extracting all information
        mu <- PCA$center # estimated centers
        Gamma <- PCA$rotation # principal axes (jth column is orthonormal eigenvector of cov(X) corresponding to jth largest eigenvalue) or 'loadings'
        Y <- PCA$x # estimated principal components of X or 'scores'
        lambda <- PCA$sdev^2 # sorted eigenvalues of Cov(X) since diag(<sorted sigma^2>) = Cov(Y) = Cov(Gamma^T (X - mu)) = Gamma^T Cov(X) Gamma = diag(<sorted lambda>)
        cumvar <- cumsum(lambda) / sum(lambda) # explained variances per first so-many principal components

        ## Return
        list(PCs = Y, cumvar = cumvar, sd = PCA$sdev, lambda = lambda, mu = mu, Gamma = Gamma)
    } else { # inverse
        ## Basics
        if(missing(mu))
            stop("'mu' (vector of centers) needs to be provided if 'inverse = TRUE'")
        if(missing(Gamma))
            stop("'Gamma' (matrix of decreasingly sorted orthonormal eigenvectors (columns)) needs to be provided if 'inverse = TRUE'")
        d.x <- ncol(x) # (old) dimension of the given/original data
        d.mu <- length(mu) # (new) dimension we want to transform to
        if(d.mu < 1)
            stop("length(mu) must be >= 1")
        d.Gamma <- dim(Gamma)
        if(d.Gamma[1] != d.mu) # check rows of Gamma
            stop("'Gamma' must have length(mu)-many rows")
        if(d.Gamma[2] < d.x) # check columns of Gamma
            stop("'Gamma' must have at least ncol(x)-many columns")

        ## Transforming back and return (we grab out as many columns from Gamma
        ## as needed, the rest is discarded)
        rep(mu, each = nrow(x)) + x %*% t(Gamma[,seq_len(d.x)]) # (n, d) + (n, k) x t((d, k))
        ## For Y = x: Y = Gamma^T (X - mu) => X = mu + Gamma Y
    }
}
