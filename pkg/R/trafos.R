### Data transformations for training/sampling #################################

##' @title Marginal Range Transformation
##' @param x (n, d)-matrix of data (typically before training or after sampling)
##' @param lower numeric or d-vector typically containing the smallest value of
##'        each column of x
##' @param upper numeric or d-vector typically containing the largest value of
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
##' @param mean numeric or d-vector of marginal sample means
##' @param sd numeric or d-vector of marginal sample standard deviations
##' @param intercept numeric or d-vector of intercepts of the linear transformations
##'        applied after applying plogis() (if inverse: before applying qlogis())
##' @param slope numeric or d-vector of slopes of the linear transformations
##'        applied after applying plogis() (if inverse: before applying qlogis())
##' @param inverse logical indicating whether a linear transformation of plogit()
##'        (the default) or qlogit() of a linear transformation of the data is
##'        applied to each component sample of x
##' @return (n, d)-matrix of marginally transformed data (linear transformations
##'         of plogis() if !inverse and qlogis() of linear transformations of the
##'         data if inverse)
##' @author Marius Hofert
##' @note On each margin, the logistic distribution/quantile function
##'       (= inverse logit/logit function) is used with location and scale chosen
##'       to match the provided (sample) mean and standard deviation.
logis_trafo <- function(x, mean = 0, sd = 1, intercept = 0, slope = 1,
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

