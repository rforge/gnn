## By Marius Hofert and Avinash Prasad

## Code to reproduce the results of Hofert, Prasad, Zhu ("Quasi-Monte Carlo for
## multivariate distributions based on generative neural networks")


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # interface to TensorFlow (numerical computation with tensors)
library(qrmtools) # for ES_np()
library(qrng) # requires at least version 0.0-6
if(packageVersion("qrng") < "0.0-6")
    stop('Consider updating via install.packages("qrng", repos = "http://R-Forge.R-project.org")')
library(copula) # considered copulas
library(gnn) # for the used GMMN models
library(latticeExtra) # for contourplot3

## Type of randomization of the QMC point sets
randomize <- "Owen" # "Owen" (scrambling) or "digital.shift"
## Note that we have to pass the integer seed as an argument to qrng::sobol()
## to guarantee reproducibility in case randomize = "Owen"

## Global training parameters
dim.hid <- 300L # dimension of the (single) hidden layer
ntrn <- 60000L # training dataset size (number of pseudo-random numbers from the copula)
nbat <- 5000L # batch size for training (number of samples per stochastic gradient step)
nepo <- 300L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
stopifnot(dim.hid >= 1, ntrn >= 1, 1 <= nbat, nbat <= ntrn, nepo >= 1)

## Other variables
firstkind <- c("normalCopula", "tCopula", "claytonCopula") # copula families for which cCopula() works
ngen <- 1000L # sample size of the generated data
B.CvM <- 100 # number of replications for Cramer-von Mises statistic
B.conv <- 25 # number of replications for convergence plots
ns <- round(2^seq(9, 18, by = 0.5)) # sequence of sample sizes for convergence plots


### 0 Auxiliary functions ######################################################

### 0.1 Computing ##############################################################

##' @title Compute Cramer-von Mises Statistic for B Replications of Samples of Size n
##'        from a Copula PRNG and a GMMN PRNG
##' @param B number of replications
##' @param n sample size of the generated (copula and GMMN) samples
##' @param copula copula object
##' @param GMMN GMMN trained on pseudo-random samples from 'copula'
##' @param randomize type or randomization used
##' @param file character string (with ending .rds) specifying the file
##'        to save the results in
##' @return (B, 2)-matrix containing the B replications of the Cramer-von Mises
##'         statistic evaluated based on generated pseudo-samples from
##'         'copula' and 'GMMN'
##' @author Marius Hofert
CvM <- function(B, n, copula, GMMN, randomize, file)
{
    if (file.exists(file)) {
        readRDS(file)
    } else {
        ## Setup
        d <- dim(copula) # copula dimension
        pb <- txtProgressBar(max = B, style = 3) # setup progress bar
        on.exit(close(pb)) # on exit, close progress bar

        ## Replications
        res <- t(sapply(seq_len(B), function(b) { # the following is independent of 'b'
            setTxtProgressBar(pb, b) # update progress bar

            ## Draw PRNs and QRNs
            U.cop.PRNG  <- pobs(rCopula(n, copula = copula)) # generate pobs of PRNs from copula
            N.PRNG <- matrix(rnorm(n * d), ncol = d) # PRNs from the prior
            U.GMMN.PRNG <- pobs(predict(GMMN, x = N.PRNG)) # generate from the GMMN PRNG
            N.QRNG <- qnorm(sobol(n, d = d, randomize = randomize, seed = b)) # QRNs from the prior
            U.GMMN.QRNG <- pobs(predict(GMMN, x = N.QRNG)) # generate from the GMMN QRNG

            ## Compute the Cramer-von Mises statistic for each of the samples
            c(gofTstat(U.cop.PRNG,  copula = copula), # CvM statistic for PRNs
              gofTstat(U.GMMN.PRNG, copula = copula), # CvM statistic for GMMN PRNs
              gofTstat(U.GMMN.QRNG, copula = copula)) # CvM statistic for GMMN QRNs
        }))

        ## Check, save and return
        stopifnot(dim(res) == c(B, 3)) # sanity check
        colnames(res) <- c("CvM.cop.PRNG", "CvM.GMMN.PRNG", "CvM.GMMN.QRNG")
        saveRDS(res, file = file)
        res
    }
}

##' @title Compute Errors for Four Test Functions for B Replications of Samples
##'        of Sizes n from a Copula PRNG and QRNG and a GMMN PRNG and QRNG
##' @param B number of replications
##' @param n vector of sample sizes of the generated (copula and GMMN) samples
##' @param copula copula object
##' @param GMMN GMMN trained on pseudo-random samples from 'copula'
##' @param randomize type or randomization used
##' @param file character string (with ending .rds) specifying the file
##'        to save the results in
##' @return (<4 test functions>, <4 RNGs>, <n>)-array containing the
##'         errors (2x mad(), 2x sd()) based on B replications of the four
##'         test functions (sum of squares, Sobol' g, 99% exceedance probability
##'         and 99% ES) evaluated for four types of RNGs (copula PRNG,
##'         GMMN PRNG, GMMN QRNG, copula QRNG) based on the sample sizes
##'         specified by 'n'.
##' @author Marius Hofert
error_test_functions <- function(B, n, copula, GMMN, randomize, file)
{
    if (file.exists(file)) {
        readRDS(file)
    } else {
        ## Setup
        d <- dim(copula) # copula dimension
        nlen <- length(n)
        pb <- txtProgressBar(max = B, style = 3) # setup progress bar
        on.exit(close(pb)) # on exit, close progress bar

        ## Iteration
        dmnms <- list("Test function" = c("Sum of squares", "Sobol' g",
                                          "Exceedance probability", "ES"),
                      "RNG" = c("PRNG", "GMMN PRNG", "GMMN QRNG", "QRNG"),
                      "n" = as.character(ns), "Replication" = 1:B)
        raw <- array(, dim = c(4, 4, nlen, B), dimnames = dmnms) # intermediate object
        for(b in seq_len(B)) { # iterate over replications (just to have 'equidistant' progress bar)
            setTxtProgressBar(pb, b) # update progress bar
            for(nind in seq_len(nlen)) { # iterate over sample sizes

                ## 0) Random number generation
                ## Draw PRNs and GMMN QRNs
                n. <- n[nind]
                U.cop.PRNG  <- rCopula(n., copula = copula) # generate pobs of PRNs from copula
                N.PRNG <- matrix(rnorm(n. * d), ncol = d) # PRNs from the prior
                U.GMMN.PRNG <- predict(GMMN, x = N.PRNG) # generate from the GMMN PRNG
                sob <- sobol(n., d = d, randomize = randomize, seed = b) # randomized Sobol' sequence; note: same seed for each 'n' (good!)
                N.QRNG <- qnorm(sob) # QRNs from the prior
                U.GMMN.QRNG <- predict(GMMN, x = N.QRNG) # generate from the GMMN QRNG
                ## Compute pseudo-observations
                U.GMMN.PRNG.pobs <- pobs(U.GMMN.PRNG)
                U.GMMN.QRNG.pobs <- pobs(U.GMMN.QRNG)
                ## If available, draw from a real QRNG
                cCopula.avail <- is(copula, "ellipCopula") || is(copula, "archmCopula") ||
                    is(copula, "rotCopula") || is(copula, "mixCopula")
                if(cCopula.avail)
                    U.cop.QRNG <- cCopula(sob, copula = copula, inverse = TRUE)

                ## 1) Compute the sum of squares test function
                ##    Note: We use the raw samples here (without pobs()) as this
                ##          test function checks the quality of the margins.
                raw[1,,nind,b] <- c(mean(sum_of_squares(U.cop.PRNG)),
                                    mean(sum_of_squares(U.GMMN.PRNG)),
                                    mean(sum_of_squares(U.GMMN.QRNG)),
                                    if(cCopula.avail) mean(sum_of_squares(U.cop.QRNG)) else NA)

                ## 2) Compute the Sobol' g test function
                raw[2,,nind,b] <- if(cCopula.avail) {
                                      c(mean(sobol_g(U.cop.PRNG,       copula = copula)),
                                        mean(sobol_g(U.GMMN.PRNG.pobs, copula = copula)),
                                        mean(sobol_g(U.GMMN.QRNG.pobs, copula = copula)),
                                        mean(sobol_g(U.cop.QRNG,       copula = copula)))
                                  } else rep(NA, 4)

                ## 3) Compute the exceedance probability over the (0.99,..,0.99) threshold
                ##    Note: Instead of Clayton, we use the survival Clayton copula here
                p <- 0.99
                trafo <- function(u) if(is(copula, "claytonCopula")) 1 - u else u
                raw[3,,nind,b] <- c(mean(exceedance(trafo(U.cop.PRNG),       q = p)),
                                    mean(exceedance(trafo(U.GMMN.PRNG.pobs), q = p)),
                                    mean(exceedance(trafo(U.GMMN.QRNG.pobs), q = p)),
                                    if(cCopula.avail)
                                        mean(exceedance(trafo(U.cop.QRNG), q = p)) else NA)

                ## 4) Compute the p-level expected shortfall
                ##    Note: Instead of Clayton, we use the survival Clayton copula here
                raw[4,,nind,b] <- c(ES_np(qnorm(trafo(U.cop.PRNG)),       level = p),
                                    ES_np(qnorm(trafo(U.GMMN.PRNG.pobs)), level = p),
                                    ES_np(qnorm(trafo(U.GMMN.QRNG.pobs)), level = p),
                                    if(cCopula.avail) ES_np(qnorm(trafo(U.cop.QRNG)), level = p))

            }
        }

        ## Compute errors, save and return
        res <- array(, dim = c(4, 4, nlen), dimnames = dmnms[1:3]) # result object
        res[1,,] <- apply(raw[1,,,], 1:2, mad) # apply mad() for fixed RNG/n combinations
        res[2,,] <- apply(raw[2,,,], 1:2, mad) # apply mad() for fixed RNG/n combinations
        res[3,,] <- apply(raw[3,,,], 1:2, sd)  # apply sd()  for fixed RNG/n combinations
        res[4,,] <- apply(raw[4,,,], 1:2, sd)  # apply sd()  for fixed RNG/n combinations
        saveRDS(res, file = file)
        res
    }
}


### 0.2 Plotting ###############################################################

##' @title Contours of the True Copula and Empirical Copulas based on
##'        GMMN-PRNG and GMMN-QRNG
##' @param copula copula object
##' @param uPRNG PRNG sample from GMMN
##' @param uQRNG QRNG sample from GMMN
##' @param file character string (with ending .pdf) specifying the PDF file
##'        to plot to or not (if not provided)
##' @param n.grid number of grid points where to evaluate contours
##' @param corner position of legend
##' @param text text of legend
##' @return nothing (plot by side-effect)
##' @author Marius Hofert
contourplot3 <- function(copula, uPRNG, uQRNG, file,
                         n.grid = 26, corner = c(0.04, 0.02),
                         text = c("True copula", "Empirical copula of GMMN PRNG",
                                  "Empirical copula of GMMN QRNG"))
{
    ## Contour plot of (true) copula
    cpTRUE <- contourplot2(copula, FUN = pCopula, region = FALSE,
                           key = list(corner = corner,
                                      lines = list(lty = 1:3, lwd = c(1, 1.3, 2.3)),
                                      text = list(text)))
    ## Grid
    u <- seq(0, 1, length.out = n.grid)
    grid <- as.matrix(expand.grid(u1 = u, u2 = u))
    ## Contour plots based on PRNG and QRNG
    cpPRNG <- contourplot2(cbind(grid, z = C.n(grid, X = uPRNG)),
                           region = FALSE, labels = FALSE, lty = 2, lwd = 1.3)
    cpQRNG <- contourplot2(cbind(grid, z = C.n(grid, X = uQRNG)),
                           region = FALSE, labels = FALSE, lty = 3, lwd = 2.3)
    ## Build plot object
    plt <- cpTRUE + cpPRNG + cpQRNG # overlaid plot
    doPDF <- hasArg(file) && is.character(file)
    if(doPDF) pdf(file = file, bg = "transparent")
    par(pty = "s")
    print(plt)
    if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file) # cropping if available
}

##' @title Plotting Rosenblatt-Transformed Bivariate Copula Samples
##' @param copula copula object
##' @param u observations (from a PRNG or QRNG)
##' @param file character string (with ending .pdf) specifying the PDF file
##'        to plot to or not (if not provided)
##' @param xlab x-axis label
##' @param ylab y-axis label
##' @return nothing (plot by side-effect)
##' @author Marius Hofert
rosenplot <- function(copula, u, file,
                      xlab = bquote(R[1]), ylab = bquote(R[2]))
{
    R <- cCopula(u, copula = copula) # Rosenblatt transform
    doPDF <- hasArg(file) && is.character(file)
    if(doPDF) pdf(file = file, bg = "transparent")
    par(pty = "s")
    plot(R, xlab = xlab, ylab = ylab)
    if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file) # cropping if available
}

##' @title Boxplot of Replications of the Cramer-von Mises Statistic
##' @param CvM (B, 3)-matrix containing the B replications of the Cramer-von Mises
##'        statistic as computed by CvM()
##' @param model.call call (as returned by quote()) for the underlying model
##' @param dim dimension of the underlying model
##' @param tau Kendall's tau of the underlying model
##' @param file character string (with ending .pdf) specifying the PDF file
##'        to plot to or not (if not provided)
##' @return nothing (plot by side-effect)
##' @author Marius Hofert
CvM_boxplot <- function(CvM, model.call, dim, tau, file)
{
    dim. <- if(length(dim) == 1) {
                as.character(dim)
            } else {
                paste0("(",paste0(dim, collapse = ", "),")")
            }
    tau. <- if(length(tau) == 1) {
                as.character(tau)
            } else {
                paste0("(",paste0(tau, collapse = ", "),")")
            }
    doPDF <- hasArg(file) && is.character(file)
    if(doPDF) pdf(file = file, bg = "transparent")
    par(pty = "s")
    boxplot(list(CvMstat[,"CvM.cop.PRNG"],
                 CvMstat[,"CvM.GMMN.PRNG"], CvMstat[,"CvM.GMMN.QRNG"]),
            names = c("Copula PRNG", "GMMN PRNG", "GMMN QRNG"),
            ylab = expression(S[n[gen]]))
    mtext(substitute(B~"replications,"~e~"copula, d ="~d*
                         ", tau ="~t, list(B = nrow(CvM), e = model.call,
                                           d = dim., t = tau.)),
          side = 4, line = 0.5, adj = 0)
    if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file) # cropping if available
}

##' @title Plot the Convergence of the Error
##' @param err (<test function>, <RNG>, <sample size>)-array
##' @param model.call call (as returned by quote()) for the underlying model
##' @param dim dimension of the underlying model
##' @param tau Kendall's tau of the underlying model
##' @param file character string (without ending .pdf) specifying the PDF file
##'        base name to plot each of the four plots to or not (if not provided)
##' @param B number of replications used for computing the error measure estimates
##' @return nothing (four plots by side-effect; one for each test function)
##' @author Marius Hofert
conv_plot <- function(err, model.call, dim, tau, filebname, B)
{
    ## Setup
    ns <- as.numeric(dimnames(err)[["n"]]) # extracting the sample sizes n
    ccoef <- function(error) abs(coef(lm(log(error) ~ log(ns)))[["log(ns)"]]) # convergence coeff.
    ## Note: error(n) = O(n^{-alpha}) => error(n) = c*n^{-alpha} => ccoef(error) = alpha
    ylabels <- rep(c(expression("Mean absolute deviation estimate,"~O(n^{-alpha})),
                     expression("Standard deviation estimate,"~O(n^{-alpha}))), each = 2)
    tfname <- c("sumofsq", "sobolg", "exceedprob99", "ES99") # test function names for PDF files
    dim. <- if(length(dim) == 1) {
                as.character(dim)
            } else {
                paste0("(",paste0(dim, collapse = ", "),")")
            }
    tau. <- if(length(tau) == 1) {
                as.character(tau)
            } else {
                paste0("(",paste0(tau, collapse = ", "),")")
            }

    ## Loop over the test functions (one plot per test function)
    for(tfind in 1:4) {
        ## Compute convergence rates (the larger alpha, the faster the convergence;
        ## for MC, alpha ~= 1/2 for sd [~= 1 for variance])
        a <- round(c(PRNG      = ccoef(err[tfind,"PRNG",]),
                     GMMN.PRNG = ccoef(err[tfind,"GMMN PRNG",]),
                     GMMN.QRNG = ccoef(err[tfind,"GMMN QRNG",]),
                     QRNG      = ccoef(err[tfind,"QRNG",])), digits = 2)

        ## Plot
        doPDF <- hasArg(filebname) && is.character(filebname)
        if(doPDF) {
            file <- paste0(filebname,"_testfun_",tfname[tfind],".pdf")
            pdf(file = file, bg = "transparent")
        }
        par(pty = "s") # square plot region
        ylim <- range(err[tfind,,], na.rm = TRUE) # determine ylim
        plot(ns, err[tfind,"PRNG",], ylim = ylim, log = "xy", type = "l", # (n, PRNG error)
             xlab = expression(n[gen]), ylab = ylabels[tfind])
        lines(ns, err[tfind,"GMMN PRNG",], type = "l", lty = 2, lwd = 1.3)
        lines(ns, err[tfind,"GMMN QRNG",], type = "l", lty = 3, lwd = 1.6)
        lines(ns, err[tfind,"QRNG",],      type = "l", lty = 4, lwd = 1.3)
        legend("bottomleft", bty = "n", lty = 1:4, lwd = c(1, 1.3, 1.6, 1.3),
               legend = as.expression(
                   c(substitute("Copula PRNG,"~alpha == a., list(a. = a["PRNG"])),
                     substitute("GMMN PRNG,"~  alpha == a., list(a. = a["GMMN.PRNG"])),
                     substitute("GMMN QRNG,"~  alpha == a., list(a. = a["GMMN.QRNG"])),
                     substitute("Copula QRNG,"~alpha == a., list(a. = a["QRNG"])))))
        mtext(substitute(B~"replications,"~e~"copula, d ="~d*
                             ", tau ="~t, list(B = B, e = model.call,
                                               d = dim., t = tau.)),
              side = 4, line = 0.5, adj = 0)
        if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file) # cropping if available
    }
}


### 1 Copulas we use ###########################################################

taus <- c(0.25, 0.5, 0.75) # Kendall's tau considered


### 1.1 d = 2 ##################################################################

d <- 2 # copula dimension

## t copulas
nu <- 4 # degrees of freedom of the t copulas
th.t <- iTau(tCopula(), tau = taus) # parameters
t.cop.d2.tau1 <- tCopula(th.t[1], dim = d, df = nu)
t.cop.d2.tau2 <- tCopula(th.t[2], dim = d, df = nu)
t.cop.d2.tau3 <- tCopula(th.t[3], dim = d, df = nu)

## Clayton copulas
th.C <- iTau(claytonCopula(), tau = taus)
C.cop.d2.tau1 <- claytonCopula(th.C[1], dim = d)
C.cop.d2.tau2 <- claytonCopula(th.C[2], dim = d)
C.cop.d2.tau3 <- claytonCopula(th.C[3], dim = d)

## Gumbel copulas
th.G <- iTau(gumbelCopula(), tau = taus)
G.cop.d2.tau1 <- gumbelCopula(th.G[1], dim = d)
G.cop.d2.tau2 <- gumbelCopula(th.G[2], dim = d)
G.cop.d2.tau3 <- gumbelCopula(th.G[3], dim = d)

## Marshall--Olkin copulas
alpha <- c(0.75, 0.60)
MO.cop.d2 <- moCopula(alpha, dim = d)

## Mixture copulas
w <- c(1/2, 1/2) # mixture weights
t.cop.d2.tau2.rot90 <- rotCopula(t.cop.d2.tau2, flip = c(TRUE, FALSE)) # t copula (tau = 0.5) rotated by 90 degrees
mix.cop.C.t90  <- mixCopula(list(C.cop.d2.tau2, t.cop.d2.tau2.rot90), w = w) # Clayton-t(90)
mix.cop.G.t90  <- mixCopula(list(G.cop.d2.tau2, t.cop.d2.tau2.rot90), w = w) # Gumbel-t(90)
mix.cop.MO.t90 <- mixCopula(list(MO.cop.d2,     t.cop.d2.tau2.rot90), w = w) # MO-t(90)


### 1.2 d = 3 ##################################################################

## Auxiliary function
nacList <- function(d, th) {
    stopifnot(length(d) == 2, d >= 1, length(th) == 3)
    list(th[1], NULL, list(list(th[2], 1:d[1]), # nesting structure
                           list(th[3], (d[1]+1):sum(d))))
}

## Nested copulas
d <- c(2, 1) # sector dimensions
NAC.d21 <- onacopulaL("Clayton", nacList = nacList(d, th = th.C)) # nested Clayton
NAG.d21 <- onacopulaL("Gumbel",  nacList = nacList(d, th = th.G)) # nested Gumbel


### 1.3 d = 5 ##################################################################

d <- 5 # copula dimension

## Basic copulas
t.cop.d5.tau2 <- tCopula(th.t[2],       dim = d, df = nu) # t copula
C.cop.d5.tau2 <- claytonCopula(th.C[2], dim = d) # Clayton copula
G.cop.d5.tau2 <- gumbelCopula(th.G[2],  dim = d) # Gumbel copula

## Nested copulas
d <- c(2, 3) # sector dimensions
NAC.d23 <- onacopulaL("Clayton", nacList = nacList(d, th = th.C)) # nested Clayton
NAG.d23 <- onacopulaL("Gumbel",  nacList = nacList(d, th = th.G)) # nested Gumbel


### 1.4 d = 10 #################################################################

d <- 10 # copula dimension

## Basic copulas
t.cop.d10.tau2 <- tCopula(th.t[2],       dim = d, df = nu) # t copula
C.cop.d10.tau2 <- claytonCopula(th.C[2], dim = d) # Clayton copula
G.cop.d10.tau2 <- gumbelCopula(th.G[2],  dim = d) # Gumbel copula

## Nested copulas
d <- c(5, 5) # sector dimensions
NAC.d55 <- onacopulaL("Clayton", nacList = nacList(d, th = th.C)) # nested Clayton
NAG.d55 <- onacopulaL("Gumbel",  nacList = nacList(d, th = th.G)) # nested Gumbel


### 2 Train the GMMNs from a PRNG of the respective copula and analyze the results

## Note: This takes long (~= 15min per copula) even if the GMMN is already trained

### 2.1 d = 2 ##################################################################

## Specifying the copula/case
cop <- t.cop.d2.tau1 # copula
model <- quote(italic(t)[4]) # model expression
mname <- paste0("t",nu) # string
tau <- taus[1] # Kendall's tau

## TODO: make the following copula work with the code thereafter (as an example of
## a more involved copula)

## Specifying the copula/case
cop <- NAC.d55 # copula
model <- "NAC" # model expression
mname <- model # string
tau <- taus[2] # Kendall's tau


### Training ###################################################################

## Generate training data and train
set.seed(271) # for reproducibility
U <- rCopula(ntrn, copula = cop) # generate training dataset from a PRNG
dim.in.out <- dim(cop) # dimension of the prior distribution fed into the GMMN
bname <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                "_nbat_",nbat,"_nepo_",nepo,"_",mname,"_tau_",tau)
GMMN <- train_once(dim = c(dim.in.out, dim.hid, dim.in.out), data = U,
                   nbat = nbat, nepo = nepo, file = paste0(bname,".rda"),
                   package = "gnn")


### Contour plot and Rosenblatt plots ##########################################
x
## Setup and data generation
bname <- paste0(mname,"_dim_",dim.in.out,"_tau_",tau) # suffix
seed <- 314
set.seed(seed) # for reproducibility
N01.prior.PRNG <- matrix(rnorm(ngen * dim.in.out), ncol = dim.in.out) # prior PRNs
N01.prior.QRNG <- qnorm(sobol(ngen, d = dim.in.out, randomize = randomize, seed = seed)) # prior QRNs
U.GMMN.PRNG <- pobs(predict(GMMN, x = N01.prior.PRNG)) # GMMN PRNs
U.GMMN.QRNG <- pobs(predict(GMMN, x = N01.prior.QRNG)) # GMMN QRNs

## Contour plot
contourplot3(cop, uPRNG = U.GMMN.PRNG, uQRNG = U.GMMN.QRNG,
             file = paste0("fig_contours_",bname,".pdf"))

## Rosenblatt plot
rosenplot(cop, u = U.GMMN.PRNG,
          file = paste0("fig_rosenblatt_",bname,".pdf"))


### Cramer-von Mises (CvM) statistic ###########################################

## Compute B.CvM replications of the CvM statistic
CvMstat <- CvM(B.CvM, n = ngen, copula = cop, GMMN = GMMN, randomize = randomize,
               file = paste0("comp_CvMstat_",bname,".rds")) # about 1min

## Boxplots
CvM_boxplot(CvMstat, model = model, dim = dim.in.out, tau = tau,
            file = paste0("fig_CvMboxplot_",bname,".pdf"))


### Test functions #############################################################

## Compute errors over B.conv replications; an (4, 4, length(ns))-array
## (<test function>, <RNG>, <sample size>)
errTFs <- error_test_functions(B.conv, n = ns, # about 13min
                               copula = cop, GMMN = GMMN, randomize = randomize,
                               file = paste0("comp_testfun_",bname,".rds"))

## Plot convergence behavior
conv_plot(errTFs, model = model, dim = dim.in.out, tau = tau,
          filebname = paste0("fig_convplot_",bname), B = B.conv)

## TODO: write one helper function to call them all and call it for each copula separately