## By Marius Hofert and Avinash Prasad

## Code to reproduce the results of Hofert, Prasad, Zhu ("Quasi-Monte Carlo for
## multivariate distributions via generative neural networks") [HPZ19] and more.


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # interface to TensorFlow (numerical computation with tensors)
library(qrmtools) # for ES_np()
if(packageVersion("qrng") < "0.0-7")
    stop('Consider updating via install.packages("qrng", repos = "http://R-Forge.R-project.org")')
library(qrng) # for sobol()
if(packageVersion("copula") < "0.999.19")
    stop('Consider updating via install.packages("copula", repos = "http://R-Forge.R-project.org")')
library(copula) # for the considered copulas
library(gnn) # for the used GMMN models
library(latticeExtra) # for contourplot3
library(parallel) # for parallel computing

## Global training parameters
package <- "gnn" # uses pre-trained NNs from 'gnn' (recommended); for retraining, use package = NULL
dim.hid <- 300L # dimension of the (single) hidden layer
ntrn <- 60000L # training dataset size (number of pseudo-random numbers from the copula)
nbat <- 5000L # batch size for training (number of samples per stochastic gradient step)
nepo <- 300L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
stopifnot(dim.hid >= 1, ntrn >= 1, 1 <= nbat, nbat <= ntrn, nepo >= 1)

## Other global parameters
firstkind <- c("normalCopula", "tCopula", "claytonCopula") # those copulas for which cCopula() works
ngen <- 1000L # sample size of the generated data
B.CvM <- 100 # number of replications for Cramer-von Mises statistic
B.conv <- 25 # number of replications for convergence plots
ns <- round(2^seq(9, 18, by = 0.5)) # sequence of sample sizes for convergence plots
ncores <- 1 # detectCores() # number of cores to be used for parallel computing
stopifnot(ncores == 1) # as of 2019, TensorFlow does not allow multicore calculations in R
## Note: See the discussion on https://stat.ethz.ch/pipermail/r-sig-hpc/2019-August/002092.html


### 0 Auxiliary functions ######################################################

### 0.1 Computing ingredients ##################################################

##' @title Compute Cramer-von Mises Statistic for B Replications of Samples of Size n
##'        from a Copula PRNG and a GMMN PRNG
##' @param B number of replications
##' @param n sample size of the generated (copula and GMMN) samples
##' @param copula copula object
##' @param GMMN GMMN trained on pseudo-random samples from 'copula'
##' @param randomize type or randomization used
##' @param file character string (with ending .rds) specifying the file
##'        to save the results in
##' @return (B, 3)-matrix containing the B replications of the Cramer-von Mises
##'         statistic evaluated based on the generated pseudo-samples from
##'         'copula', the GMMN PRNs and the GMMN QRNs
##' @author Marius Hofert
##' @note Could have added QRNGs based on cCopula() for those copulas available
CvM <- function(B, n, copula, GMMN, randomize, file)
{
    if (file.exists(file)) {
        readRDS(file)
    } else {
        ## Setup
        GMMNmod <- GMMN[["model"]]
        d <- dim(copula) # copula dimension

        ## Auxiliary function
        aux <- function(b) { # the following is independent of 'b'
            ## Draw PRNs and QRNs
            U.cop.PRNG  <- pobs(rCopula(n, copula = copula)) # generate pobs of PRNs from copula
            N.PRNG <- matrix(rnorm(n * d), ncol = d) # PRNs from the prior
            U.GMMN.PRNG <- pobs(predict(GMMNmod, x = N.PRNG)) # generate from the GMMN PRNG
            N.QRNG <- qnorm(sobol(n, d = d, randomize = randomize, seed = b)) # QRNs from the prior
            U.GMMN.QRNG <- pobs(predict(GMMNmod, x = N.QRNG)) # generate from the GMMN QRNG
            ## Compute the Cramer-von Mises statistic for each of the samples
            c(gofTstat(U.cop.PRNG,  copula = copula), # CvM statistic for PRNs
              gofTstat(U.GMMN.PRNG, copula = copula), # CvM statistic for GMMN PRNs
              gofTstat(U.GMMN.QRNG, copula = copula)) # CvM statistic for GMMN QRNs
        }

        ## Replications
        RNGkind("L'Ecuyer-CMRG") # switch PRNG to CMRG (for reproducible parallel computing)
        raw <- mclapply(seq_len(B), function(b) aux(b), mc.cores = ncores)
        RNGkind("Mersenne-Twister") # switch back to default RNG
        res <- simplify2array(raw) # or, here: matrix(unlist(raw), ncol = 3, byrow = TRUE)

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
##' @note Could have made this faster by only generating the largest sample
##'       and then take sub-samples (not so for sobol()-calls, though)
error_test_functions <- function(B, n, copula, GMMN, randomize, file)
{
    if (file.exists(file)) {
        readRDS(file)
    } else {
        ## Setup
        GMMNmod <- GMMN[["model"]]
        d <- dim(copula) # copula dimension
        nlen <- length(n)
        dmnms <- list("Test function" = c("Sum of squares", "Sobol' g",
                                          "Exceedance probability", "ES"),
                      "RNG" = c("PRNG", "GMMN PRNG", "GMMN QRNG", "QRNG"),
                      "n" = as.character(ns)) # dimnames of result object

        ## Helper function for the big iteration
        aux <- function(b) {
            ## Result object of aux()
            r <- array(, dim = c(4, 4, nlen), dimnames = dmnms) # test function, type of RNG, sample size

            ## Loop
            for(nind in seq_len(nlen)) { # iterate over sample sizes
                ## 0) Random number generation
                ## Draw PRNs
                n. <- n[nind]
                U.cop.PRNG  <- rCopula(n., copula = copula) # generate pobs of PRNs from copula
                ## Draw GMMN PRNs
                N.PRNG <- matrix(rnorm(n. * d), ncol = d) # PRNs from the prior
                U.GMMN.PRNG <- predict(GMMNmod, x = N.PRNG) # generate from the GMMN PRNG
                U.GMMN.PRNG.pobs <- pobs(U.GMMN.PRNG) # compute pseudo-observations
                ## Draw GMMN QRNs
                sob <- sobol(n., d = d, randomize = randomize, seed = b) # randomized Sobol' sequence; note: same seed for each 'n' (good!)
                N.QRNG <- qnorm(sob) # QRNs from the prior
                U.GMMN.QRNG <- predict(GMMNmod, x = N.QRNG) # generate from the GMMN QRNG
                U.GMMN.QRNG.pobs <- pobs(U.GMMN.QRNG) # compute pseudo-observations
                ## If available in analytical form, draw from a real QRNG
                cCopula.inverse.avail <- is(copula, "normalCopula") || is(copula, "tCopula") ||
                    is(copula, "claytonCopula")
                if(cCopula.inverse.avail)
                    U.cop.QRNG <- cCopula(sob, copula = copula, inverse = TRUE)

                ## 1) Compute the sum of squares test function
                ##    Note: We use the raw samples here (without pobs()) as this
                ##          test function checks the quality of the margins.
                r[1,,nind] <- c(mean(sum_of_squares(U.cop.PRNG)),
                                mean(sum_of_squares(U.GMMN.PRNG)),
                                mean(sum_of_squares(U.GMMN.QRNG)),
                                if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                    mean(sum_of_squares(U.cop.QRNG)) else NA)

                ## 2) Compute the Sobol' g test function
                ##    Note: Requires cCopula() to be available (which holds for all 'copula'
                ##          we call this function with except NACs)
                cCopula.avail <- !is(copula, "outer_nacopula")
                r[2,,nind] <- if(cCopula.avail) {
                                  c(mean(sobol_g(U.cop.PRNG,       copula = copula)),
                                    mean(sobol_g(U.GMMN.PRNG.pobs, copula = copula)),
                                    mean(sobol_g(U.GMMN.QRNG.pobs, copula = copula)),
                                    if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                        mean(sobol_g(U.cop.QRNG, copula = copula)) else NA)
                              } else rep(NA, 4)

                ## 3) Compute the exceedance probability over the (0.99,..,0.99) threshold
                ##    Note: Instead of Clayton, we use the survival Clayton copula here
                p <- 0.99
                trafo <- function(u) if(is(copula, "claytonCopula")) 1 - u else u
                r[3,,nind] <- c(mean(exceedance(trafo(U.cop.PRNG),       q = p)),
                                mean(exceedance(trafo(U.GMMN.PRNG.pobs), q = p)),
                                mean(exceedance(trafo(U.GMMN.QRNG.pobs), q = p)),
                                if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                    mean(exceedance(trafo(U.cop.QRNG), q = p)) else NA)

                ## 4) Compute the p-level expected shortfall
                ##    Note: Instead of Clayton, we use the survival Clayton copula here
                r[4,,nind] <- c(ES_np(qnorm(trafo(U.cop.PRNG)),       level = p),
                                ES_np(qnorm(trafo(U.GMMN.PRNG.pobs)), level = p),
                                ES_np(qnorm(trafo(U.GMMN.QRNG.pobs)), level = p),
                                if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                    ES_np(qnorm(trafo(U.cop.QRNG)), level = p) else NA)
            }

            ## Return of aux()
            r
        } # aux()

        ## Replications
        RNGkind("L'Ecuyer-CMRG") # switch PRNG to CMRG (for reproducible parallel computing)
        raw <- mclapply(seq_len(B), function(b) aux(b), mc.cores = ncores)
        RNGkind("Mersenne-Twister") # switch back to default RNG
        res. <- simplify2array(raw) # convert list of 3-arrays to 4-array
        names(dimnames(res.))[4] <- "Replication" # update name of dimnames
        dimnames(res.)[[4]] <- 1:B  # update dimnames

        ## Compute errors, save and return
        res <- array(, dim = c(4, 4, nlen), dimnames = dmnms[1:3]) # result object
        res[1,,] <- apply(res.[1,,,], 1:2, mad) # apply mad() for fixed RNG, n combinations
        res[2,,] <- apply(res.[2,,,], 1:2, mad) # apply mad() for fixed RNG, n combinations
        res[3,,] <- apply(res.[3,,,], 1:2, sd)  # apply sd()  for fixed RNG, n combinations
        res[4,,] <- apply(res.[4,,,], 1:2, sd)  # apply sd()  for fixed RNG, n combinations
        saveRDS(res, file = file)
        res
    }
}

##' @title Human-readable Elapsed Time
##' @param expr R expression to evaluate and time
##' @param string character string to be printed before the time
##' @return human-readable rounded elapsed time
##' @author Marius Hofert
gettime <- function(expr, string = "=> Overall done in")
{
    st <- system.time(expr)[["elapsed"]]
    res <- if(st < 60) {
        paste0(sprintf("%.1f", round(st, 1)),"s") # time in seconds
    } else if(st < 3600) {
        paste0(sprintf("%.1f", round(st/60, 1)),"m") # time in minutes
    } else paste0(sprintf("%.1f", round(st/3600, 1)),"h") # time in hours
    cat(paste(string,res,"\n"))
}


### 0.2 Plotting ###############################################################

##' @title Contours of the True Copula and Empirical Copulas based on
##'        GMMN PRNG and GMMN QRNG
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
    cpTRUE <- contourplot2(copula, FUN = pCopula, region = FALSE, col = "gray50",
                           key = list(corner = corner,
                                      lines = list(lty = 1:3, lwd = c(1, 1.3, 2.3),
                                                   col = c("gray50", "black", "black")),
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
    if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file)
}

##' @title Plotting Rosenblatt-Transformed Bivariate Copula Samples
##' @param copula copula object
##' @param u observations (from a PRNG, QRNG, GMMN PRNG or GMMN QRNG)
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
    if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file)
}

##' @title Scatter Plots
##' @param u observations (from a PRNG, QRNG, GMMN PRNG or GMMN QRNG)
##' @param file character string (with ending .pdf) specifying the PDF file
##'        to plot to or not (if not provided)
##' @return nothing (plot by side-effect)
##' @author Marius Hofert
scatterplot <- function(u, file)
{
    doPDF <- hasArg(file) && is.character(file)
    if(doPDF) pdf(file = file, bg = "transparent")
    par(pty = "s")
    if(ncol(u) == 2) {
        plot(u, xlab = quote(U[1]), ylab = quote(U[2]))
    } else pairs2(u)
    if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file)
}

##' @title Boxplot of Replications of the Cramer-von Mises Statistic
##' @param CvM (B, 3)-matrix containing the B replications of the Cramer-von Mises
##'        statistic as computed by CvM()
##' @param dim dimension of the underlying model
##' @param model call (as returned by quote()) for the underlying model including tau(s)
##' @param file character string (with ending .pdf) specifying the PDF file
##'        to plot to or not (if not provided)
##' @return nothing (plot by side-effect)
##' @author Marius Hofert
CvM_boxplot <- function(CvM, dim, model, file)
{
    dim. <- if(length(dim) == 1) {
                as.character(dim)
            } else {
                paste0("(",paste0(dim, collapse = ", "),")")
            }
    doPDF <- hasArg(file) && is.character(file)
    if(doPDF) pdf(file = file, bg = "transparent", width = 7.4, height = 7.4)
    par(pty = "s")
    boxplot(list(CvM[,"CvM.cop.PRNG"], CvM[,"CvM.GMMN.PRNG"], CvM[,"CvM.GMMN.QRNG"]),
            names = c("Copula PRNG", "GMMN PRNG", "GMMN QRNG"),
            ylab = expression(S[n[gen]]))
    mtext(substitute(B~"replications, d ="~d*","~m,
                     list(B = nrow(CvM), d = dim., m = model)),
          side = 4, line = 0.5, adj = 0)
    if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file) # cropping if available
}

##' @title Plot the Convergence of the Error
##' @param err (<test function>, <RNG>, <sample size>)- or (<RNG>, <sample size>)-array
##' @param dim dimension of the underlying model
##' @param model call (as returned by quote()) for the underlying model including tau(s)
##' @param file character string (without ending .pdf) specifying the PDF file
##'        base name to plot each of the four plots to or not (if not provided)
##' @param B number of replications used for computing the error measure estimates
##' @return nothing (one or four plots by side-effect; one for each test function)
##' @author Marius Hofert
convergence_plot <- function(err, dim, model, filebname, B)
{
    ## Setup
    ns <- as.numeric(dimnames(err)[["n"]]) # extracting the sample sizes n
    ccoef <- function(error) { # convergence coefficient
        res <- tryCatch(lm(log(error) ~ log(ns)), error = function(e) e)
        if(is(res, "simpleError")) NA else -coef(res)[["log(ns)"]]
    }
    ## Note: error(n) = O(n^{-alpha}) => error(n) = c*n^{-alpha} => ccoef(error) = alpha
    ld <- length(dim(err))
    if(ld == 3) { # for all test functions
        ylabels <- rep(c(expression("Mean absolute deviation estimate,"~O(n[gen]^{-alpha})),
                         expression("Standard deviation estimate,"~O(n[gen]^{-alpha}))), each = 2)
        tfname <- c("sumofsq", "sobolg", "exceedprob99", "ES99") # test function names for PDF files
        tfnum <- 4 # number of test functions
    } else if(ld == 2) { # for the ES test function
        ylabels <- expression("Standard deviation estimate,"~O(n[gen]^{-alpha}))
        tfname <- "ES99"
        tfnum <- 1
    }
    dim. <- if(length(dim) == 1) {
                as.character(dim)
            } else {
                paste0("(",paste0(dim, collapse = ", "),")")
            }

    ## Loop over the test functions (one plot per test function)
    for(ind in 1:tfnum) {
        err. <- if(tfnum == 1) err else err[ind,,]

        ## Compute convergence rates (the larger alpha, the faster the convergence;
        ## for MC, alpha ~= 1/2 for sd [~= 1 for variance])
        a <- round(c(PRNG      = ccoef(err.["PRNG",]),
                     GMMN.QRNG = ccoef(err.["GMMN QRNG",]),
                     QRNG      = ccoef(err.["QRNG",])), digits = 2)
        if(all(is.na(a))) next # no plot; happens for Sobol' g test function and copulas without available cCopula()
        ## Now it could still happen that a["QRNG"] is NA (omit this case from the plot then)

        ## Plot
        doPDF <- hasArg(filebname) && is.character(filebname)
        if(doPDF) {
            file <- paste0(filebname,"_testfun_",tfname[ind],".pdf")
            pdf(file = file, bg = "transparent", width = 7.4, height = 7.4)
        }
        par(pty = "s")
        ylim <- range(err.[,], na.rm = TRUE)
        lgnd <- as.expression(
            c(substitute("Copula PRNG,"~alpha == a., list(a. = a["PRNG"])),
              substitute("GMMN QRNG,"~  alpha == a., list(a. = a["GMMN.QRNG"])),
              if(!is.na(a["QRNG"]))
                  substitute("Copula QRNG,"~alpha == a., list(a. = a["QRNG"]))))
        plot(ns, err.["PRNG",], ylim = ylim, log = "xy", type = "l",
             xlab = expression(n[gen]), ylab = ylabels[ind])
        lines(ns, err.["GMMN QRNG",], type = "l", lty = 2, lwd = 1.3)
        if(!is.na(a["QRNG"])) {
            lines(ns, err.["QRNG",], type = "l", lty = 3, lwd = 1.6)
            legend("bottomleft", bty = "n", lty = 1:3, lwd = c(1, 1.3, 1.6), legend = lgnd)
        } else {
            legend("bottomleft", bty = "n", lty = 1:2, lwd = c(1, 1.3), legend = lgnd)
        }
        mtext(substitute(B.~"replications, d ="~d*","~m,
                         list(B. = B, d = dim., m = model)),
              side = 4, line = 0.5, adj = 0)
        if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file)
    }
}


### 0.3 Wrappers reproducing all results #######################################

##' @title Results of the Main Part of the Paper
##' @param copula copula object
##' @param name character string (copula and taus together) for trained GMMNs,
##'        computed CvM statistics and test functions (.rds objects) as well as
##'        in corresponding boxplot and convergence plot figures
##' @param model call containing a model string used in boxplots and
##'        convergence plots; not used if CvM.testfun = FALSE
##' @param randomize type or randomization used
##' @param CvM.testfun logical indicating whether the CvM statistics and
##'        the test functions are evaluated
##' @return nothing (computes results by side-effect)
##' @author Marius Hofert
##' @note Uses global variables to keep number of arguments small
main <- function(copula, name, model, randomize, CvM.testfun = TRUE)
{
    ## 1 Training ##############################################################

    ## Generate training data
    set.seed(271) # for reproducibility
    U <- rCopula(ntrn, copula = copula) # generate training dataset from a PRNG
    ## Train
    dim.in.out <- dim(copula) # = dimension of the prior distribution fed into the GMMN
    NNname <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                     "_nbat_",nbat,"_nepo_",nepo,"_",name,".rda")
    GNN <- GMMN_model(c(dim.in.out, dim.hid, dim.in.out)) # model setup
    cat(paste0("=> Starting training (unless pre-trained). "))
    gettime(GMMN <- train_once(GNN, data = U,
                               batch.size = nbat, nepoch = nepo,
                               file = NNname, package = package),
            string = "Done in") # training and saving

    ## 2 Contour/Rosenblatt plots or scatter plots #############################

    ## Setup and data generation
    GMMNmod <- GMMN[["model"]]
    bname <- paste0("dim_",dim.in.out,"_",name) # suffix
    seed <- 314
    set.seed(seed) # for reproducibility
    N01.prior.PRNG <- matrix(rnorm(ngen * dim.in.out), ncol = dim.in.out) # prior PRNs
    N01.prior.QRNG <- qnorm(sobol(ngen, d = dim.in.out, randomize = randomize, seed = seed)) # prior QRNs
    U.GMMN.PRNG <- pobs(predict(GMMNmod, x = N01.prior.PRNG)) # GMMN PRNs
    U.GMMN.QRNG <- pobs(predict(GMMNmod, x = N01.prior.QRNG)) # GMMN QRNs

    ## Contour, Rosenblatt and scatter plots
    cat("=> Computing contour, Rosenblatt and scatter plots\n")
    if(dim.in.out == 2 && !grepl("MO", x = name)) { # rosenblatt() not available for copulas involving MO (MO itself or mixtures)
        contourplot3(copula, uPRNG = U.GMMN.PRNG, uQRNG = U.GMMN.QRNG,
                     file = paste0("HPZ19_fig_contours_",bname,".pdf"))
        rosenplot(copula, u = U.GMMN.QRNG,
                  file = paste0("HPZ19_fig_rosenblatt_",bname,".pdf"))
    }
    ## Scatter plots
    if(dim.in.out <= 3) { # for larger dimensions, one doesn't see much anyways
        lst <- list(PRNG = U[seq_len(ngen),], GMMN.PRNG = U.GMMN.PRNG, GMMN.QRNG = U.GMMN.QRNG)
        nms <- c("PRNG", "GMMN_PRNG", "GMMN_QRNG")
        for(i in seq_along(lst))
            scatterplot(lst[[i]], file = paste0("HPZ19_fig_scatter_",bname,"_",nms[i],".pdf"))
    }

    ## 3 Cramer-von Mises (CvM) statistics and test functions ##################

    if(CvM.testfun) {

        ## Compute string of tau(s) to be displayed in plots
        tau.str <- if(is(copula, "outer_nacopula")) {
                       th <- sort(unique(as.vector(nacPairthetas(copula))))
                       taus <- sapply(th, function(th.)
                           tau(archmCopula(copula@copula@name, param = th.)))
                       paste0("(",paste0(taus, collapse = ", "),")")
                   } else as.character(tau(copula))
        model. <- substitute(m.*","~~tau==t., list(m. = model, t. = tau.str)) # model and taus

        ## 3.1 CVM statistics ##################################################

        ## Compute B.CvM replications of the CvM statistic
        cat("=> Starting to compute Cramer-von Mises statistics. ")
        gettime(CvMstat <- CvM(B.CvM, n = ngen, copula = copula, GMMN = GMMN,
                               randomize = randomize,
                               file = paste0("HPZ19_res_CvMstat_",bname,".rds")),
                string = "Done in")

        ## Boxplots
        CvM_boxplot(CvMstat, dim = dim.in.out, model = model.,
                    file = paste0("HPZ19_fig_CvMboxplot_",bname,".pdf"))

        ## 3.2 Test functions ##################################################

        ## Compute errors over B.conv replications; an (4, 4, length(ns))-array
        ## (<test function>, <RNG>, <sample size>)
        cat("=> Starting to compute errors for test functions. ")
        gettime(errTFs <- error_test_functions(B.conv, n = ns,
                                               copula = copula, GMMN = GMMN,
                                               randomize = randomize,
                                               file = paste0("HPZ19_res_testfun_",bname,".rds")),
                string = "Done in")

        ## Plot convergence behavior
        convergence_plot(errTFs, dim = dim.in.out, model = model.,
                         filebname = paste0("HPZ19_fig_convergence_",bname), B = B.conv)

    }
}

##' @title Results of the Appendix of the Paper
##' @param copula copula object
##' @param name character string (copula and taus together) for trained GMMNs,
##'        computed CvM statistics and test functions (.rds objects) as well as
##'        in corresponding boxplot and convergence plot figures
##' @param model call containing a model string used in boxplots and
##'        convergence plots; not used if CvM.testfun = FALSE
##' @param randomize type or randomization used
##' @return nothing (computes results by side-effect)
##' @author Marius Hofert
##' @note Uses global variables to keep number of arguments small
appendix <- function(copula, name, model, randomize)
{
    ## 1 Training ##############################################################

    ## Generate training data
    set.seed(271) # for reproducibility
    U <- rCopula(ntrn, copula = copula) # generate training dataset from a PRNG
    ## Train
    dim.in.out <- dim(copula) # = dimension of the prior distribution fed into the GMMN
    NNname <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                     "_nbat_",nbat,"_nepo_",nepo,"_",name,".rda")
    GNN <- GMMN_model(c(dim.in.out, dim.hid, dim.in.out)) # model setup
    cat(paste0("=> Starting training (unless pre-trained). "))
    gettime(GMMN <- train_once(GNN, data = U,
                               batch.size = nbat, nepoch = nepo,
                               file = NNname, package = package),
            string = "Done in") # training and saving

    ## 2 Expected shortfall test function ######################################

    GMMNmod <- GMMN[["model"]]
    bname <- paste0("dim_",dim.in.out,"_",name) # suffix
    file <- paste0("HPZ19_res_testfun_",bname,"_digital_shift.rds")
    res <- if (file.exists(file)) {
        readRDS(file)
    } else {
        ## Setup
        d <- dim(copula)
        n <- ns
        B <- B.conv
        nlen <- length(n)
        dmnms <- list("RNG" = c("PRNG", "GMMN PRNG", "GMMN QRNG", "QRNG"),
                      "n" = as.character(ns)) # dimnames of result object

        ## Helper function for the big iteration
        aux <- function(b) { # iterate over replications
            ## Result object of aux()
            r <- array(, dim = c(4, nlen), dimnames = dmnms) # type of RNG, sample size

            ## Loop
            for(nind in seq_len(nlen)) { # iterate over sample sizes
                ## 0) Random number generation
                ## Draw PRNs
                n. <- n[nind]
                U.cop.PRNG  <- rCopula(n., copula = copula) # generate pobs of PRNs from copula
                ## Draw GMMN PRNs
                N.PRNG <- matrix(rnorm(n. * d), ncol = d) # PRNs from the prior
                U.GMMN.PRNG <- predict(GMMNmod, x = N.PRNG) # generate from the GMMN PRNG
                U.GMMN.PRNG.pobs <- pobs(U.GMMN.PRNG) # compute pseudo-observations
                ## Draw GMMN QRNs
                sob <- sobol(n., d = d, randomize = randomize, seed = b) # randomized Sobol' sequence; note: same seed for each 'n' (good!)
                N.QRNG <- qnorm(sob) # QRNs from the prior
                U.GMMN.QRNG <- predict(GMMNmod, x = N.QRNG) # generate from the GMMN QRNG
                U.GMMN.QRNG.pobs <- pobs(U.GMMN.QRNG) # compute pseudo-observations
                ## If available in analytical form, draw from a real QRNG
                cCopula.inverse.avail <- is(copula, "normalCopula") || is(copula, "tCopula") ||
                    is(copula, "claytonCopula")
                if(cCopula.inverse.avail)
                    U.cop.QRNG <- cCopula(sob, copula = copula, inverse = TRUE)
                ## 1) Compute the p-level expected shortfall
                ##    Note: Instead of Clayton, we use the survival Clayton copula here
                p <- 0.99
                trafo <- function(u) if(is(copula, "claytonCopula")) 1 - u else u
                r[,nind] <- c(ES_np(qnorm(trafo(U.cop.PRNG)),       level = p),
                              ES_np(qnorm(trafo(U.GMMN.PRNG.pobs)), level = p),
                              ES_np(qnorm(trafo(U.GMMN.QRNG.pobs)), level = p),
                              if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                  ES_np(qnorm(trafo(U.cop.QRNG)), level = p) else NA)
            }

            ## Return of aux()
            r
        } # aux()

        ## Replications
        cat("=> Starting to compute errors for test functions. ")
        RNGkind("L'Ecuyer-CMRG") # switch PRNG to CMRG (for reproducible parallel computing)
        gettime(raw <- mclapply(seq_len(B), function(b) aux(b), mc.cores = ncores),
                string = "Done in")
        RNGkind("Mersenne-Twister") # switch back to default RNG
        res. <- simplify2array(raw) # convert list of 2-arrays to 3-array
        names(dimnames(res.))[3] <- "Replication" # update name of dimnames
        dimnames(res.)[[3]] <- 1:B  # update dimnames

        ## Compute errors, save and return
        res <- apply(res., 1:2, sd) # apply sd() for fixed RNG, n combinations
        dimnames(res) <- dmnms[1:2]
        saveRDS(res, file = file)
        res
    }

    ## Plot convergence behavior
    tau.str <- if(is(copula, "outer_nacopula")) {
                   th <- sort(unique(as.vector(nacPairthetas(copula))))
                   taus <- sapply(th, function(th.)
                       tau(archmCopula(copula@copula@name, param = th.)))
                   paste0("(",paste0(taus, collapse = ", "),")")
               } else as.character(tau(copula))
    model. <- substitute(m.*","~~tau==t., list(m. = model, t. = tau.str)) # model and taus
    convergence_plot(res, dim = dim.in.out, model = model.,
                     filebname = paste0("HPZ19_fig_convergence_",bname,
                                        "_digital_shift"), B = B.conv)
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
alpha <- c(0.75, 0.60) # implies tau = 0.5
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
    if(d[1] == 1) {
        list(th[1], 1, list(list(th[2], 1 + 1:d[2])))
    } else if(d[2] == 1) {
        list(th[1], d[1]+1, list(list(th[2], 1:d[1])))
    } else {
        list(th[1], NULL, list(list(th[2], 1:d[1]),
                           list(th[3], (d[1]+1):sum(d))))
    }
}

## Nested copulas
d <- c(2, 1) # sector dimensions
NC.d21 <- onacopulaL("Clayton", nacList = nacList(d, th = th.C)) # nested Clayton
NG.d21 <- onacopulaL("Gumbel",  nacList = nacList(d, th = th.G)) # nested Gumbel


### 1.3 d = 5 ##################################################################

d <- 5 # copula dimension

## Basic copulas
t.cop.d5.tau2 <- tCopula(th.t[2],       dim = d, df = nu) # t copula
C.cop.d5.tau2 <- claytonCopula(th.C[2], dim = d) # Clayton copula
G.cop.d5.tau2 <- gumbelCopula(th.G[2],  dim = d) # Gumbel copula

## Nested copulas
d <- c(2, 3) # sector dimensions
NC.d23 <- onacopulaL("Clayton", nacList = nacList(d, th = th.C)) # nested Clayton
NG.d23 <- onacopulaL("Gumbel",  nacList = nacList(d, th = th.G)) # nested Gumbel


### 1.4 d = 10 #################################################################

d <- 10 # copula dimension

## Basic copulas
t.cop.d10.tau2 <- tCopula(th.t[2],       dim = d, df = nu) # t copula
C.cop.d10.tau2 <- claytonCopula(th.C[2], dim = d) # Clayton copula
G.cop.d10.tau2 <- gumbelCopula(th.G[2],  dim = d) # Gumbel copula

## Nested copulas
d <- c(5, 5) # sector dimensions
NC.d55 <- onacopulaL("Clayton", nacList = nacList(d, th = th.C)) # nested Clayton
NG.d55 <- onacopulaL("Gumbel",  nacList = nacList(d, th = th.G)) # nested Gumbel


### 2 Train the GMMNs from a PRNG of the respective copula and analyze the results

## Timings are on a 13" MacBook Pro (2018) without training. Overall, with
## pre-trained NNs, this runs in a bit less than 11h (via R CMD BATCH HPZ19.R,
## for example).


### 2.1 Main part of the paper #################################################

## Copulas from Section 1.1 above
gettime(main(t.cop.d2.tau1, name = paste0("t",nu,"_tau_",taus[1]), # ~= 3.6s
             model = quote(italic(t)[4]), randomize = "Owen",
             CvM.testfun = FALSE))
gettime(main(t.cop.d2.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 16.0m
             model = quote(italic(t)[4]), randomize = "Owen"))
gettime(main(t.cop.d2.tau3, name = paste0("t",nu,"_tau_",taus[3]), # ~= 2.3s
             model = quote(italic(t)[4]), randomize = "Owen",
             CvM.testfun = FALSE))
gettime(main(C.cop.d2.tau1, name = paste0("C","_tau_",taus[1]), # ~= 2.3s
             model = quote(Clayton), randomize = "Owen",
             CvM.testfun = FALSE))
gettime(main(C.cop.d2.tau2, name = paste0("C","_tau_",taus[2]), # ~= 7.0m
             model = quote(Clayton), randomize = "Owen"))
gettime(main(C.cop.d2.tau3, name = paste0("C","_tau_",taus[3]), # ~= 2.4s
             model = quote(Clayton), randomize = "Owen",
             CvM.testfun = FALSE))
gettime(main(G.cop.d2.tau1, name = paste0("G","_tau_",taus[1]), # ~= 2.5s
             model = quote(Gumbel), randomize = "Owen",
             CvM.testfun = FALSE))
gettime(main(G.cop.d2.tau2, name = paste0("G","_tau_",taus[2]), # ~= 12.3m
             model = quote(Gumbel), randomize = "Owen"))
gettime(main(G.cop.d2.tau3, name = paste0("G","_tau_",taus[3]), # ~= 2.6s
             model = quote(Gumbel), randomize = "Owen",
             CvM.testfun = FALSE))
gettime(main(MO.cop.d2, name = paste0("MO_",paste0(alpha,collapse = "_")), # ~= 2.2s
             model = quote(MO), randomize = "Owen",
             CvM.testfun = FALSE)) # argument 'model' actually not used here
gettime(main(mix.cop.C.t90, name = "eqmix_C_tau_0.5_rot90_t4_tau_0.5", # ~= 3.4s
             model = quote(Clayton-italic(t)[4](90)), randomize = "Owen",
             CvM.testfun = FALSE)) # argument 'model' actually not used here
gettime(main(mix.cop.G.t90,  name = "eqmix_G_tau_0.5_rot90_t4_tau_0.5", # ~= 3.4s
             model = quote(Gumbel-italic(t)[4](90)), randomize = "Owen",
             CvM.testfun = FALSE)) # argument 'model' actually not used here
gettime(main(mix.cop.MO.t90, # ~= 2.4s
             name = paste0("eqmix_MO_",paste0(alpha,collapse = "_"),"_rot90_t4_tau_0.5"),
             model = quote(MO-italic(t)[4](90)), randomize = "Owen",
             CvM.testfun = FALSE)) # argument 'model' actually not used here

## Copulas from Section 1.2 above
gettime(main(NC.d21, name = paste0("NC21_tau_",paste0(taus[1:2], collapse = "_")), # ~= 5.5m
             model = quote("(2,1)-nested Clayton"), randomize = "Owen"))
gettime(main(NG.d21, name = paste0("NG21_tau_",paste0(taus[1:2], collapse = "_")), # ~= 5.3m
             model = quote("(2,1)-nested Gumbel"), randomize = "Owen"))

## Copulas from Section 1.3 above
gettime(main(t.cop.d5.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 58.1m
             model = quote(italic(t)[4]), randomize = "Owen"))
gettime(main(C.cop.d5.tau2, name = paste0("C","_tau_",taus[2]), # ~= 10.6m
             model = quote(Clayton), randomize = "Owen"))
gettime(main(G.cop.d5.tau2, name = paste0("G","_tau_",taus[2]), # ~= 33.6m
             model = quote(Gumbel), randomize = "Owen"))
gettime(main(NC.d23, name = paste0("NC23_tau_",paste0(taus, collapse = "_")), # ~= 7.0m
             model = quote("(2,3)-nested Clayton"), randomize = "Owen"))
gettime(main(NG.d23, name = paste0("NG23_tau_",paste0(taus, collapse = "_")), # ~= 6.6m
             model = quote("(2,3)-nested Gumbel"), randomize = "Owen"))

## Copulas from Section 1.4 above
gettime(main(t.cop.d10.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 4.4h
             model = quote(italic(t)[4]), randomize = "Owen"))
gettime(main(C.cop.d10.tau2, name = paste0("C","_tau_",taus[2]), # ~= 13.8m
             model = quote(Clayton), randomize = "Owen"))
gettime(main(G.cop.d10.tau2, name = paste0("G","_tau_",taus[2]), # ~= 1.1h
             model = quote(Gumbel), randomize = "Owen"))
gettime(main(NC.d55, name = paste0("NC55_tau_",paste0(taus, collapse = "_")), # ~= 9.1m
             model = quote("(5,5)-nested Clayton"), randomize = "Owen"))
gettime(main(NG.d55, name = paste0("NG55_tau_",paste0(taus, collapse = "_")), # ~= 8.7m
             model = quote("(5,5)-nested Gumbel"), randomize = "Owen"))


### 2.2 Appendix ###############################################################

## Note: No .rds will be written in this case (just the plots generated directly)

## Row 1
gettime(appendix(t.cop.d2.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 7.4m
                 model = quote(italic(t)[4]), randomize = "digital.shift"))
gettime(appendix(t.cop.d5.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 12.7m
                 model = quote(italic(t)[4]), randomize = "digital.shift"))
gettime(appendix(t.cop.d10.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 21.3m
                 model = quote(italic(t)[4]), randomize = "digital.shift"))
## Row 2
gettime(appendix(C.cop.d2.tau2, name = paste0("C","_tau_",taus[2]), # ~= 6.1m
                 model = quote(Clayton), randomize = "digital.shift"))
gettime(appendix(C.cop.d5.tau2, name = paste0("C","_tau_",taus[2]), # ~= 7.4m
                 model = quote(Clayton), randomize = "digital.shift"))
gettime(appendix(C.cop.d10.tau2, name = paste0("C","_tau_",taus[2]), # ~= 9.1m
                 model = quote(Clayton), randomize = "digital.shift"))
## Row 3
gettime(appendix(G.cop.d2.tau2, name = paste0("G","_tau_",taus[2]), # ~= 6.4m
                 model = quote(Gumbel), randomize = "digital.shift"))
gettime(appendix(G.cop.d5.tau2, name = paste0("G","_tau_",taus[2]), # ~= 7.4m
                 model = quote(Gumbel), randomize = "digital.shift"))
gettime(appendix(G.cop.d10.tau2, name = paste0("G","_tau_",taus[2]), # ~= 8.9m
                 model = quote(Gumbel), randomize = "digital.shift"))
## Row 4
gettime(appendix(NG.d21, name = paste0("NG21_tau_",paste0(taus[1:2], collapse = "_")), # ~= 7.1m
                 model = quote("(2,1)-nested Gumbel"), randomize = "digital.shift"))
gettime(appendix(NG.d23, name = paste0("NG23_tau_",paste0(taus, collapse = "_")), # ~= 8.0m
                 model = quote("(2,3)-nested Gumbel"), randomize = "digital.shift"))
gettime(appendix(NG.d55, name = paste0("NG55_tau_",paste0(taus, collapse = "_")), # ~= 9.5m
                 model = quote("(5,5)-nested Gumbel"), randomize = "digital.shift"))
