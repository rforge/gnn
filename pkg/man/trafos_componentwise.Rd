\name{trafos_componentwise}
\alias{range_trafo}
\alias{logis_trafo}
\title{Data Transformations for Training or Sampling}
\description{
  Transformations applied to each marginal component sample to map given data
  to a different range.
}
\usage{
range_trafo(x, lower, upper, inverse = FALSE)
logis_trafo(x, mean = 0, sd = 1, slope = 1, intercept = 0, inverse = FALSE)
}
\arguments{
  \item{x}{\eqn{(n, d)}-matrix of data (typically before training or after sampling).}
  \item{lower}{value or \eqn{d}-vector typically
    containing the smallest value of each column of \code{x}.}
  \item{upper}{value or \eqn{d}-vector typically
    containing the largest value of each column of \code{x}.}
  \item{mean}{value or \eqn{d}-vector.}
  \item{sd}{value or \eqn{d}-vector.}
  \item{slope}{value or \eqn{d}-vector of slopes
    of the linear transformations applied after applying
    \code{\link{plogis}()} (before applying \code{\link{qlogis}()} if
    \code{inverse = TRUE}).}
  \item{intercept}{value or \eqn{d}-vector of intercepts
    of the linear transformations applied after applying
    \code{\link{plogis}()} (before applying \code{\link{qlogis}()} if
    \code{inverse = TRUE}).}
  \item{inverse}{\code{\link{logical}} indicating whether the inverses
    of the respective transformations are to be computed (typically
    used after generating data from a neural network trained on
    data transformed with the respective transformation and
    \code{inverse = FALSE}).}
}
\value{
  An object as \code{x} containing the componentwise transformed data.
}
\author{Marius Hofert}
\examples{
## Generate data
n <- 100
set.seed(271)
x <- cbind(rnorm(n), (1-runif(n))^(-1/2)-1) # normal and Pareto(2) margins
plot(x)

## Range transformation
ran <- apply(x, 2, range) # column j = range of the jth column of x
x.ran <- range_trafo(x, lower = ran[1,], upper = ran[2,]) # marginally transform to [0,1]
plot(x.ran) # => now range [0,1] but points a bit clustered around small y-values
x. <- range_trafo(x.ran, lower = ran[1,], upper = ran[2,], inverse = TRUE) # transform back
stopifnot(all.equal(x., x)) # check

## Logistic transformation
x.logis <- logis_trafo(x) # marginally transform to [0,1] via plogis()
plot(x.logis) # => y-range is [1/2, 1] which can be harder to train
x. <- logis_trafo(x.logis, inverse = TRUE) # transform back
stopifnot(all.equal(x., x)) # check

## Logistic transformation with scaling to all of [0,1] in the second coordinate
x.logis.scale <- logis_trafo(x, slope = 2, intercept = -1)
plot(x.logis.scale) # => now y-range is scaled to [0,1]
x. <- logis_trafo(x.logis.scale, slope = 2, intercept = -1, inverse = TRUE) # transform back
stopifnot(all.equal(x., x)) # check

## Logistic transformation with sample mean and standard deviation and then
## transforming the range to [0,1] with a range transformation (note that
## slope = 2, intercept = -1 would not help here as the y-range is not [1/2, 1])
mu <- colMeans(x)
sig <- apply(x, 2, sd)
x.logis.fit <- logis_trafo(x, mean = mu, sd = sig) # marginally plogis(, location, scale)
plot(x.logis.fit) # => y-range is not [1/2, 1] => use range transformation
ran <- apply(x.logis.fit, 2, range)
x.logis.fit.ran <- range_trafo(x.logis.fit, lower = ran[1,], upper = ran[2,])
plot(x.logis.fit.ran) # => now y-range is [1/2, 1]
x. <- logis_trafo(range_trafo(x.logis.fit.ran, lower = ran[1,], upper = ran[2,],
                              inverse = TRUE),
                  mean = mu, sd = sig, inverse = TRUE) # transform back
stopifnot(all.equal(x., x)) # check

## Note that for heavy-tailed data, plogis() can fail to stay inside (0,1)
## even with adapting to sample mean and standard deviation. We now present
## a case where we see that using a fitted logistic distribution function
## is *just* good enough to numerically keep the data inside (0,1).
set.seed(271)
x <- cbind(rnorm(n), (1-runif(n))^(-2)-1) # normal and Pareto(1/2) margins
plot(x) # => heavy-tailed in y-coordinate
## Transforming with standard logistic distribution function
x.logis <- logis_trafo(x)
stopifnot(any(x.logis[,2] == 1))
## => There is value numerically indistinguishable from 1 to which applying
##    the inverse transform will lead to Inf
stopifnot(any(is.infinite(logis_trafo(x.logis, inverse = TRUE))))
## Now adapt the logistic distribution to share the mean and standard deviation
## with the data
mu <- colMeans(x)
sig <- apply(x, 2, sd)
x.logis.scale <- logis_trafo(x, mean = mu, sd = sig)
stopifnot(all(x.logis.scale[,2] != 1)) # => no values equal to 1 anymore

## Alternatively, log() the data first, thus working with a log-logistic
## distribution as transformation
lx <- cbind(x[,1], log(x[,2])) # 2nd coordinate only
lmu <- c(mu[1], mean(lx[,2]))
lsig <- c(sig[1], sd(lx[,2]))
x.llogis <- logis_trafo(lx, mean = lmu, sd = lsig)
x. <- logis_trafo(x.llogis, mean = lmu, sd = lsig, inverse = TRUE)
x.. <- cbind(x.[,1], exp(x.[,2])) # undo log()
stopifnot(all.equal(x.., x))
}
\keyword{manip}
