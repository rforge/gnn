## By Marius Hofert

## Basic tests of QMC based on GMMNs

## Note: If TensorFlow was installed in a virtual environment as described on
##       https://www.tensorflow.org/install/pip#system-install, then this needs
##       to be activated ('source .../bin/activate/') before this script can be run;
##       the path to the virtual environment can be found via echo $VIRTUAL_ENV


## Packages
library(tensorflow) # R package 'tensorflow'; load *before* gnn
library(gnn) # load *after* tensorflow; otherwise the wrong 'train' is used (which produces an error)
library(qrng)

## Checks
if(!TensorFlow_available()) q() # as training and evaluation below would fail
stopifnot(packageVersion("qrng") >= "0.0-7")

## Training data
d <- 2 # bivariate case
P <- matrix(0.9, nrow = d, ncol = d); diag(P) <- 1 # correlation matrix
A <- t(chol(P)) # Cholesky factor A s.t. AA^T = P
ntrn <- 60000 # training data sample size
set.seed(271)
Z <- matrix(rnorm(ntrn * d), ncol = d) # N(0,1)^d samples
X <- t(A %*% t(Z)) # N(0,P) samples
X. <- abs(X) # |X|
U <- apply(X., 2, rank) / (ntrn + 1) # pseudo-observations of |X|

## Plot a subsample
m <- 2000 # subsample size
opar <- par(pty = "s")
plot(U[1:m,], xlab = expression(U[1]), ylab = expression(U[2])) # visual check (PRNG)

## Define the model and 'train' it
dim <- c(d, 300, d) # dimensions of the input, hidden and output layers
GMMN.mod <- GMMN_model(dim) # define the GMMN model
nbat <- 500 # batch size = number of samples per gradient step (=> 120x gradient steps per epoch)
nepo <- 10 # number of epochs = number of times the training data is shuffled/revisited
GMMN <- train(GMMN.mod, data = U, batch.size = nbat, nepoch = nepo)
## Note:
## - Obviously, in a real-world application, batch.size and nepoch
##   should be (much) larger (e.g., batch.size = 5000, nepoch = 300).
## - The above training is not reproducible (due to keras).

## Evaluate GMMN based on prior sample (already roughly picks up the shape)
set.seed(271)
N.prior <- matrix(rnorm(m * d), ncol = d) # sample from the prior distribution
V <- predict(GMMN[["model"]], x = N.prior) # feedforward through GMMN

## Joint plot
layout(t(1:2))
plot(U[1:m,], xlab = expression(U[1]), ylab = expression(U[2]), cex = 0.2) # training subsample
plot(V,       xlab = expression(V[1]), ylab = expression(V[2]), cex = 0.2) # GMMN PRNG

## Generate and plot QRNG samples
V. <- predict(GMMN[["model"]], x = qnorm(sobol(m, d = d, randomize = "Owen", seed = 271)))
plot(U[1:m,], xlab = expression(U[1]), ylab = expression(U[2]), cex = 0.2) # training subsample
plot(V.,      xlab = expression(V[1]), ylab = expression(V[2]), cex = 0.2) # GMMN QRNG
layout(1)
par(opar)
