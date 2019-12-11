## By Marius Hofert and Avinash Prasad

## Minimal working example for QMC based on GMMNs. Due to the dependency on
## TensorFlow, this code cannot be turned into a vignette for 'gnn' (even if
## pretrained neural networks from 'gnn' are used), but is used for testing
## if TensorFlow is available.

## The aim of this project was to construct a general-purpose quasi-random number
## generator (QRNG) for multivariate distributions in order to estimate an expectation
## $E(\Psi(\mathbf{X}))$ for some measurable $\Psi:\mathbf{R}^d\to\mathbf{R}$ and
## a random vector $\mathbf{X}=(X_1,\dots,X_d)\sim F_{\mathbf{X}}$ with copula $C$
## and margins $F_{X_1},\dots,F_{X_d}$. Our approach utilizes generative moment
## matching networks (GMMNs) to conceptually construct an approximation
## $\mathbf{Y}$ to $\mathbf{X}$ and then uses the randomized quasi-Monte Carlo
## (RQMC) estimator $\frac{1}{n}\sum_{i=1}^n \Psi(\mathbf{Y}_i)$, where
## where $\mathbf{Y}_i=f_{\mathbf{\hat{\theta}}}\circ
## F_{\mathbf{Z}}^{-1}(\mathbf{\tilde{v}}_i)$, $i=1,\dots,n$, for
## $f_{\mathbf{\hat{\theta}}}$ being the trained GMMN (with estimated parameters
## $\mathbf{\hat{\theta}}$),
## $F_{\mathbf{Z}}^{-1}(\mathbf{u})=(F_{Z_1}^{-1}(u_1),\dots,F_{Z_p}^{-1}(u_p))$
## and $\{\mathbf{\tilde{v}}_1,\dots,\mathbf{\tilde{v}}_n\}$ being a $p$-dimensional
## randomized quasi-Monte Carlo point set (such as randomized Sobol').

## Note that Cambou, Hofert and Lemieux (2016, "Quasi-random numbers for copula
## models") recently utilized the copula concept for constructing QRNGs for very
## specific $F_{\mathbf{X}}$.  Our goal here is to construct QRNGs for much more
## general $F_{\mathbf{X}}$. Training can be done based on any PRNG (which are
## available for virtually any high-dimensional model used in practice, in
## contrast to QRNGs) which makes this variance reduction approach widely applicable.
## For more details and many more examples, we refer to Hofert, Prasad and
## Zhu (2018) and focus here on the main steps and how to achieve them in R.

## We start by loading the R packages we need and specifying some variables which
## will remain fixed for all examples considered.


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # interface to TensorFlow (numerical computation with tensors)
if(packageVersion("qrng") < "0.0-7")
    stop('Consider updating via install.packages("qrng", repos = "http://R-Forge.R-project.org")')
library(qrng) # for sobol()
if(packageVersion("copula") < "0.999.19")
    stop('Consider updating via install.packages("copula", repos = "http://R-Forge.R-project.org")')
library(copula) # for the considered copulas
library(gnn) # for the used GMMN models
library(latticeExtra) # for contourplot3()
library(qrmtools) # For ES_np()

## Global training parameters
package <- "gnn" # uses pre-trained NNs from 'gnn' (recommended); for retraining, use package = NULL
dim.hid <- 300L # dimension of the (single) hidden layer
ntrn <- 60000L # training dataset size (number of pseudo-random numbers from the copula)
nbat <- 5000L # batch size for training (number of samples per stochastic gradient step)
nepo <- 300L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
stopifnot(dim.hid >= 1, ntrn >= 1, 1 <= nbat, nbat <= ntrn, nepo >= 1)


### 0 Auxiliary functions ######################################################

##' @title Compute Cramer-von Mises Statistic for B Replications of Samples of Size n
##'        from a Copula PRNG, GMMN PRNG and GMMN QRNG
##' @param B number of replications
##' @param n sample size of the generated (copula and GMMN) samples
##' @param copula copula object
##' @param GMMN GMMN trained on pseudo-random samples from 'copula'
##' @param randomize type or randomization used
##' @param file character string (with ending .rda) specifying the file
##'        to save the results in
##' @param name name under which the object is saved in 'file'
##' @param package name of the package from which to read the object; if NULL
##'        (the default) the current working directory is used.
##' @return (B, 3)-matrix containing the B replications of the Cramer-von Mises
##'         statistics evaluated based on pseudo-random samples from
##'         'copula', GMMN PRNs and GMMN QRNs.
##' @author Marius Hofert
##' @note This is an adapted version of the same function in the demo GMMN_QMC_paper
CvM <- function(B, n, copula, GMMN, randomize, file, name = rm_ext(basename(file)),
                package = NULL)
{
    if(exists_rda(file, names = name, package = package)) {
        read_rda(name, file = file, package = package)
    } else {
        ## Setup
        GMMNmod <- GMMN[["model"]]
        d <- dim(copula) # copula dimension

        ## Auxiliary function
        aux <- function(b) { # the following is independent of 'b'
            ## Draw copula PRNs
            U.cop.PRNG <- pobs(rCopula(n, copula = copula)) # generate pobs of PRNs from copula
            ## Draw GMMN PRNs
            N.PRNG <- matrix(rnorm(n * d), ncol = d) # PRNs from the prior distribution
            U.GMMN.PRNG <- pobs(predict(GMMNmod, x = N.PRNG)) # generate from the GMMN PRNG
            ## Draw GMMN QRNs
            N.QRNG <- qnorm(sobol(n, d = d, randomize = randomize, seed = b)) # QRNs from prior
            U.GMMN.QRNG <- pobs(predict(GMMNmod, x = N.QRNG)) # generate from the GMMN QRNG
            ## Compute the Cramer-von Mises statistic for each of the samples
            c(gofTstat(U.cop.PRNG,  copula = copula), # CvM statistic for copula PRNs
              gofTstat(U.GMMN.PRNG, copula = copula), # CvM statistic for GMMN PRNs
              gofTstat(U.GMMN.QRNG, copula = copula)) # CvM statistic for GMMN QRNs
        }

        ## Replications
        raw <- lapply(seq_len(B), function(b) {
            cat(paste0("Working on replication ",b," of ",B,"\n"))
            aux(b)
        })
        res <- t(simplify2array(raw))

        ## Check, save and return
        stopifnot(dim(res) == c(B, 3)) # sanity check
        colnames(res) <- c("CvM.cop.PRNG", "CvM.GMMN.PRNG", "CvM.GMMN.QRNG")
        save_rda(res, file = file, names = name)
        res
    }
}

##' @title Compute Expected Shortfall Estimates of the Aggregate Loss for B Replications
##'        of Samples of Size n from a Copula PRNG, GMMN PRNG, GMMN QRNG and Copula QRNG
##' @param B number of replications
##' @param n sample size of the generated (copula and GMMN) samples
##' @param copula copula object
##' @param GMMN GMMN trained on pseudo-random samples from 'copula'
##' @param randomize type or randomization used
##' @return (B, 4)-matrix containing the B replications of the computed expected
##'         shortfalls (at confidence level 99%) evaluated based on
##'         pseudo-random samples from 'copula', GMMN PRNs, GMMN QRNs and
##'         copula QRNs.
ES99 <- function(B, n, copula, GMMN, randomize)
{
    ## Setup
    GMMNmod <- GMMN[["model"]]
    d <- dim(copula) # copula dimension

    ## Auxiliary function
    aux <- function(b) { # the following is independent of 'b'
        ## Draw copula PRNs
        U.cop.PRNG <- rCopula(n, copula = copula) # (aggregate) loss (PRNG)
        ## Draw GMMN PRNs
        N.PRNG <- matrix(rnorm(n * d), ncol = d) # PRNs from the prior distribution
        U.GMMN.PRNG <- pobs(predict(GMMNmod, x = N.PRNG)) # generate from the GMMN PRNG
        ## Draw GMMN QRNs
        N.QRNG <- qnorm(sobol(n, d = d, randomize = randomize, seed = b)) # QRNs from prior
        U.GMMN.QRNG <- pobs(predict(GMMNmod, x = N.QRNG)) # generate from the GMMN QRNG
        ## Draw copula QRNs (available here)
        U.cop.QRNG <- cCopula(sobol(n, d = d, randomize = randomize, seed = b),
                              copula = copula, inverse = TRUE)
        ## Compute ES_0.99 for each of the samples
        p <- 0.99 # confidence level
        c(ES_np(qnorm(U.cop.PRNG),  level = p), # ES_0.99 for copula PRNs
          ES_np(qnorm(U.GMMN.PRNG), level = p), # ES_0.99 for GMMN PRNs
          ES_np(qnorm(U.GMMN.QRNG), level = p), # ES_0.99 for GMMN QRNs
          ES_np(qnorm(U.cop.QRNG),  level = p)) # ES_0.99 for copula QRNs
    }

    ## Replications
    raw <- lapply(seq_len(B), function(b) aux(b))
    res <- t(simplify2array(raw))

    ## Check and return
    stopifnot(dim(res) == c(B, 4)) # sanity check
    colnames(res) <- c("ES99.cop.PRNG", "ES99.GMMN.PRNG",
                       "ES99.GMMN.QRNG", "ES99.cop.QRNG")
    res
}

##' @title Standard Deviations of Expected Shortfall Estimates of the Aggregate Loss
##'        for B Replications of Samples of Size n from a Copula PRNG, GMMN PRNG,
##'        GMMN QRNG and Copula QRNG
##' @param B number of replications
##' @param ns sample sizes of the generated (copula and GMMN) samples
##' @param copula copula object
##' @param GMMN GMMN trained on pseudo-random samples from 'copula'
##' @param randomize type or randomization used
##' @param file character string (with ending .rda) specifying the file
##'        to save the results in
##' @param name name under which the object is saved in 'file'
##' @param package name of the package from which to read the object; if NULL
##'        (the default) the current working directory is used.
ES99_sd <- function(B, ns, copula, GMMN, randomize, file,
                    name = rm_ext(basename(file)), package = NULL)
{
    if(exists_rda(file, names = name, package = package)) {
        sds <- read_rda(name, file = file, package = package)
    } else {
        nslen <- length(ns)
        sds <- t(sapply(seq_len(nslen), function(i) {
            cat(paste0("Working on sample size number ",i," of ",nslen,"\n"))
            apply(ES99(B, n = ns[i], copula = copula, GMMN = GMMN, randomize = randomize),
                  2, sd)
        }))
        rownames(sds) <- ns
        colnames(sds) <- c("Sd.ES99.cop.PRNG",  "Sd.ES99.GMMN.PRNG",
                           "Sd.ES99.GMMN.QRNG", "Sd.ES99.cop.QRNG")
        save_rda(sds, file = file, names = name)
    }
    sds
}


### 1 A bivariate mixture copula example #######################################

## As a bivariate example copula, we consider a mixture between a Clayton copula
## and a $t_4$ copula which is rotated by 90 degrees (with equal mixture weights).


### 1.1 Training ###############################################################

## Define a Clayton-t(90) copula (a mixture of a Clayton and 90 degree rotated t copula)
d <- 2 # dimension of the output layer of the GMMN (= copula dimension)
tau <- 0.5 # Kendall's tau of the Clayton and the (rotated) t copula
nu <- 4 # degrees of freedom of the (rotated) t copula
C.cop <- claytonCopula(iTau(claytonCopula(), tau = tau), dim = d) # Clayton copula
t.cop <- tCopula(iTau(tCopula(), tau = tau), dim = d, df = nu) # t copula
t90.cop <- rotCopula(t.cop, flip = c(TRUE, FALSE)) # t copula rotated by 90 degrees
mix.cop.C.t90 <- mixCopula(list(C.cop, t90.cop), w = c(1/2, 1/2)) # the mixture copula

## Next, we generate the training dataset from a PRNG from this mixture copula.
set.seed(271) # for reproducibility
U <- rCopula(ntrn, copula = mix.cop.C.t90) # training dataset

## We now set up the GMMN and train it based on the training dataset and a sample
## of equal size from the prior distribution. The idea here is that the GMMN thus
## learns the map from a sample from the prior distribution to the training dataset.
dim.in.out <- dim(mix.cop.C.t90) # = dimension of the prior distribution for the GMMN
NNname <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                 "_nbat_",nbat,"_nepo_",nepo,"_eqmix_C_tau_",tau,"_rot90_t",
                 nu,"_tau_",tau,".rda")
GNN <- GMMN_model(c(dim.in.out, dim.hid, dim.in.out)) # model setup
GMMN <- train_once(GNN, data = U, batch.size = nbat, nepoch = nepo,
                   file = NNname, package = package) # training/saving/loading


### 1.2 Visualizing PRNs, GMMN PRNs and GMMN QRNs ##############################

## After training of the GMMN (to the training dataset of the mixture copula),
## we can use it to generate pseudo-random numbers (PRNs) from this mixture copula
## if we feed the GMMN with data from the prior (independent $\mathrm{N}(0,1)$).
## Moreover, and this is a rather surprising result, we can use the GMMN to
## generate quasi-random numbers (QRNs) from the mixture copula when feeding it
## with QRNs from the prior (easily obtained); note that the variance-reduction
## effect is shown later.
ngen <- 1000L # sample size of the generated data
## Sample from the prior distribution (PRNs and QRNs)
N.PRNG <- matrix(rnorm(ngen * dim.in.out), ncol = dim.in.out) # N(0,1) PRNs
N.QRNG <- qnorm(sobol(ngen, d = dim.in.out, randomize = "Owen")) # N(0,1) QRNs
## Generate data from the fitted GMMN (for pobs(), see Hofert, Prasad and Zhu (2018))
U.GMMN.PRNG <- pobs(predict(GMMN[["model"]], x = N.PRNG))
U.GMMN.QRNG <- pobs(predict(GMMN[["model"]], x = N.QRNG))

## Let us now use these samples to produce plots similar to the top row of Figure 7
## of Hofert, Prasad and Zhu (2018) consisting of a pseudo-random sample from the
## mixture copula, a pseudo-random sample from the trained GMMN and a quasi-random
## sample from the trained GMMN.
layout(t(1:3)) # 1 x 3 layout
opar <- par(pty = "s") # square plots
plot(U[1:ngen,], cex = 0.2, xlab = bquote(U[1]), ylab = bquote(U[2]))
mtext("Copula PRNG sample", cex = 0.6, side = 4, line = 0.5, adj = 0)
plot(U.GMMN.PRNG, cex = 0.2, xlab = bquote(U[1]), ylab = bquote(U[2]))
mtext("GMMN PRNG sample", cex = 0.6, side = 4, line = 0.5, adj = 0)
plot(U.GMMN.QRNG, cex = 0.2, xlab = bquote(U[1]), ylab = bquote(U[2]))
mtext("GMMN QRNG sample", cex = 0.6, side = 4, line = 0.5, adj = 0)
par(opar) # restore graphical parameters
layout(1) # restore layout

## Note that Hofert, Prasad and Zhu (2018) even consider mixtures with singular
## components and the GMMN picks up such null sets well.


### 1.3 Visually assessing the accuracy of the GMMN samples ####################

## We now briefly consider a visual assessment of whether the generated samples
## from the trained GMMN are indeed samples from the mixture copula as specified.
## To this end, we compute the Rosenblatt transformation of the GMMN PRNs and
## GMMN QRNs.

## Rosenblatt transform of the Clayton-t(90) mixture copula from GMMN PRNG samples
R.GMMN.PRNG <- cCopula(U.GMMN.PRNG, copula = mix.cop.C.t90)
## Rosenblatt transform of the Clayton-t(90) mixture copula from GMMN QRNG samples
R.GMMN.QRNG <- cCopula(U.GMMN.QRNG, copula = mix.cop.C.t90)

## If the training worked well, the Rosenblatt transformed samples should produce
## random numbers from $\mathrm{U}(0,1)^2$. Let us now visually verify this.
layout(t(1:2)) # 1 x 2 layout
opar <- par(pty = "s") # square plots
plot(R.GMMN.PRNG, cex = 0.2, xlab = bquote(R[1]), ylab = bquote(R[2]))
mtext("Rosenblatt-transformed GMMN PRNG sample", cex = 0.6,
      side = 4, line = 0.5, adj = 0)
plot(R.GMMN.QRNG, cex = 0.2, xlab = bquote(R[1]), ylab = bquote(R[2]))
mtext("Rosenblatt-transformed GMMN QRNG sample", cex = 0.6,
      side = 4, line = 0.5, adj = 0)
par(opar) # restore graphical parameters
layout(1) # restore layout
## The right-hand side plot is similar to the one on the bottom-left of Figure 4 of
## Hofert, Prasad and Zhu (2018); see the top-left plot in this reference for a
## comparison of the level curves of the empirical copulas based on the GMMN PRNs
## and based on the GMMN QRNs with the level curves of the true copula. The level
## curves of the empirical copula based on the GMMN QRNs follow the level curves of
## the true copula more closely than the ones of the empirical copula based on the
## GMMN PRNs.


### 3 A higher-dimensional $t$ copula example ##################################

### 3.1 Training ###############################################################

## In this section, we consider a five-dimensional $t$ copula (for which a QRNG
## exists, see Cambou, Hofert and Lemieux (2016); this allows for a comparison).
## We start by defining the copula.

## Define a t_4 copula
dim.in.out <- 5 # dimension of the input/output layer of the GMMN (= copula dimension)
tau <- 0.5 # Kendall's tau
nu <- 4 # degrees of freedom
t.cop <- tCopula(iTau(tCopula(), tau = tau), dim = dim.in.out, df = nu) # t copula
m <- quote(italic(t)[4]) # model string

## We now generate the training dataset from a PRNG from this copula (as before).
set.seed(271) # for reproducibility
U <- rCopula(ntrn, copula = t.cop) # training dataset

## Next, we set up the GMMN and train it based on the training dataset and a
## sample of equal size from the prior distribution.
dim.in.out <- dim(t.cop) # = dimension of the prior distribution fed into the GMMN
NNname <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                 "_nbat_",nbat,"_nepo_",nepo,"_t",nu,"_tau_",tau,".rda")
GNN <- GMMN_model(c(dim.in.out, dim.hid, dim.in.out)) # model setup
GMMN <- train_once(GNN, data = U, batch.size = nbat, nepoch = nepo,
                   file = NNname, package = package) # training/saving/loading


### 3.2 Assessing the accuracy of the GMMN samples #############################

## To assess the accuracy of the GMMN samples in higher dimensions, we generate 100
## replications of Cramer-von Mises statistics for pseudo-random copula samples,
## the GMMN PRNs and the GMMN QRNs.
file <- paste0("GMMN_QMC_res_CvMstats_GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,
               "_ntrn_",ntrn,"_nbat_",nbat,"_nepo_",nepo,"_t4_tau_",tau,".rda")
CvMstat <- CvM(100, n = ngen, copula = t.cop, GMMN = GMMN, randomize = "Owen",
               file = file, package = package)

## We can now compare the computed Cramer-von Mises statistics for the different
## sampling approaches with a box plot; see the top-left plot in Figure 8 of
## Hofert, Prasad and Zhu (2018).
opar <- par(pty = "s")
boxplot(list(CvMstat[,"CvM.cop.PRNG"], CvMstat[,"CvM.GMMN.PRNG"],
             CvMstat[,"CvM.GMMN.QRNG"]),
        names = c("Copula PRNG", "GMMN PRNG", "GMMN QRNG"),
        ylab = expression(S[n[gen]]))
mtext(substitute(B~"replications, d ="~d.*","~m.*","~tau==t.,
                 list(B = nrow(CvMstat), d. = dim.in.out, m. = m, t. = tau(t.cop))),
      side = 4, line = 0.5, adj = 0)
par(opar)


### 3.3 Variance reduction capability ##########################################

## In this section we demonstrate the efficiency of our GMMN-based randomized
## quasi-Monte Carlo (RQMC) estimator by analyzing its variance reduction when
## estimating the risk measure expected shortfall of an aggregate loss at
## confidence level $0.99$. To this end, we first define two auxiliary functions
## and then compute the standard deviations of four expected shortfall estimators
## (one based on a copula PRNG, one based on the GMMN PRNG, the GMMN QRNG and a
## copula QRNG) for different sample sizes.

## For various sample sizes, compute 25 replications of expected shortfall estimates
## and their standard deviation
file <- paste0("GMMN_QMC_res_ES99_sd_GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,
               "_ntrn_",ntrn,"_nbat_",nbat,"_nepo_",nepo,"_t4_tau_",tau,".rda")
B <- 25 # number of replications per computed standard deviation
ns <- round(2^seq(9, 18, by = 0.5)) # sample sizes n for which ES sd() is estimated
ES99sd <- ES99_sd(B, ns = ns, copula = t.cop, GMMN = GMMN,
                  randomize = "Owen", file = file, package = package)

## Compute regression coefficients as an approximation of the convergence rates.
## Note: error(n) = O(n^{-alpha}) => error(n) = c*n^{-alpha} => ccoef(error) = alpha
ccoef <- function(error) { # convergence coefficient
    res <- tryCatch(lm(log(error) ~ log(ns)), error = function(e) e)
    if(is(res, "simpleError")) NA else -coef(res)[["log(ns)"]]
}
alpha <- apply(ES99sd, 2, ccoef)
names(alpha) <- c("Cop.PRNG", "GMMN.PRNG", "GMMN.QRNG", "Cop.QRNG")
a <- round(alpha, digits = 2)

## We now plot the computed standard deviations of the four estimators as
## functions of the considered sample sizes $n_{\text{gen}}=\{2^9,2^{9.5},
## \dots,2^{18}\}$; compare with the top middle plot in Figure 13 of
## Hofert, Prasad and Zhu (2018).
opar <- par(pty = "s") # square plots
ran <- range(ES99sd) # plot range
plot(ns, ES99sd[,"Sd.ES99.cop.PRNG"], ylim = ran, type = "l", log = "xy",
     xlab = expression(n[gen]),
     ylab = expression("Standard deviation estimate,"~O(n[gen]^{-alpha})))
lines(ns, ES99sd[,"Sd.ES99.GMMN.PRNG"], lty = 2, lwd = 1.3)
lines(ns, ES99sd[,"Sd.ES99.GMMN.QRNG"], lty = 3, lwd = 1.6)
lines(ns, ES99sd[,"Sd.ES99.cop.QRNG"],  lty = 4, lwd = 1.3)
legend("topright", bty = "n", lty = 1:4,
       legend = as.expression(
           c(substitute("Copula PRNG,"~alpha == a., list(a. = a["Cop.PRNG"])),
             substitute("GMMN PRNG,"~  alpha == a., list(a. = a["GMMN.PRNG"])),
             substitute("GMMN QRNG,"~  alpha == a., list(a. = a["GMMN.QRNG"])),
             substitute("Copula QRNG,"~alpha == a., list(a. = a["Cop.QRNG"])))))
mtext(substitute(B.~"replications, d ="~d*","~m.*","~tau==t.,
                 list(B. = B, d = dim.in.out, m. = m, t. = tau(t.cop))),
      side = 4, line = 0.5, adj = 0)
par(opar) # restore graphical parameters
