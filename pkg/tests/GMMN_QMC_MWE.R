## By Marius Hofert

## Basic tests of QMC based on GMMNs

library(gnn)

## Check (too restrictive as the OS-level TensorFlow installation will not catch
## TensorFlow installations done differently; see the stackoverflow link)
checkCMD <- tryCatch(checkTF <- system("pip list | grep tensorflow",
                                       intern = TRUE, ignore.stderr = TRUE),
                     error = function(e) e) # see https://stackoverflow.com/questions/38549253/how-to-find-which-version-of-tensorflow-is-installed-in-my-system
TFisInstalled <- !is(checkCMD, "simpleError") && length(checkTF) > 0 &&
    grepl("tensorflow", checkTF[[1]])
doTest <- TFisInstalled && # OS-level TensorFlow
    require(tensorflow) && # tensorflow package
    require(qrng) && packageVersion("qrng") >= "0.0-7"

if(!doTest) q()


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
