## By Marius Hofert and Avinash Prasad

## Code to reproduce the results of Hofert, Prasad, Zhu ("Quasi-Monte Carlo for
## multivariate distributions based on generative neural networks") and more.


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # interface to TensorFlow (numerical computation with tensors)
library(qrmtools) # for ES_np()
library(qrng) # for sobol()
if(packageVersion("qrng") < "0.0-7")
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
##' @return (B, 2)-matrix containing the B replications of the Cramer-von Mises
##'         statistic evaluated based on generated pseudo-samples from
##'         'copula' and 'GMMN'
##' @author Marius Hofert
##' @note Could have added QRNGs based on cCopula() for those copulas available
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
##' @param verbose logical indicating whether verbose output is given
##' @return (<4 test functions>, <4 RNGs>, <n>)-array containing the
##'         errors (2x mad(), 2x sd()) based on B replications of the four
##'         test functions (sum of squares, Sobol' g, 99% exceedance probability
##'         and 99% ES) evaluated for four types of RNGs (copula PRNG,
##'         GMMN PRNG, GMMN QRNG, copula QRNG) based on the sample sizes
##'         specified by 'n'.
##' @author Marius Hofert
##' @note Could have made this faster by only generating the largest sample
##'       and then take sub-samples (not so for sobol()-calls, though)
error_test_functions <- function(B, n, copula, GMMN, randomize, file, verbose = FALSE)
{
    if (file.exists(file)) {
        readRDS(file)
    } else {
        ## Setup
        d <- dim(copula) # copula dimension
        nlen <- length(n)
        if(!verbose) {
            pb <- txtProgressBar(max = B, style = 3) # setup progress bar
            on.exit(close(pb)) # on exit, close progress bar
        }

        ## Iteration
        dmnms <- list("Test function" = c("Sum of squares", "Sobol' g",
                                          "Exceedance probability", "ES"),
                      "RNG" = c("PRNG", "GMMN PRNG", "GMMN QRNG", "QRNG"),
                      "n" = as.character(ns), "Replication" = 1:B)
        raw <- array(, dim = c(4, 4, nlen, B), dimnames = dmnms) # intermediate object
        for(b in seq_len(B)) { # iterate over replications (just to have 'equidistant' progress bar)
            if(!verbose) {
                setTxtProgressBar(pb, b) # update progress bar
            } else {
                cat(paste0("Working on replication ",b," of ",B,"\n"))
            }
            for(nind in seq_len(nlen)) { # iterate over sample sizes
                if(verbose)
                    cat(paste0("Working on the ",nind,"th sample size of ",nlen,"\n"))

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
                ## If available in analytical form, draw from a real QRNG
                cCopula.inverse.avail <- is(copula, "normalCopula") || is(copula, "tCopula") ||
                    is(copula, "claytonCopula")
                if(cCopula.inverse.avail)
                    U.cop.QRNG <- cCopula(sob, copula = copula, inverse = TRUE)

                ## 1) Compute the sum of squares test function
                ##    Note: We use the raw samples here (without pobs()) as this
                ##          test function checks the quality of the margins.
                raw[1,,nind,b] <- c(mean(sum_of_squares(U.cop.PRNG)),
                                    mean(sum_of_squares(U.GMMN.PRNG)),
                                    mean(sum_of_squares(U.GMMN.QRNG)),
                                    if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                        mean(sum_of_squares(U.cop.QRNG)) else NA)
                if(verbose) cat("=> Test function sum_of_squares() done\n")

                ## 2) Compute the Sobol' g test function
                ##    Note: Requires cCopula() to be available (which holds for all 'copula'
                ##          we call this function with except NACs)
                cCopula.avail <- !is(copula, "outer_nacopula")
                raw[2,,nind,b] <- if(cCopula.avail) {
                                      c(mean(sobol_g(U.cop.PRNG,       copula = copula)),
                                        mean(sobol_g(U.GMMN.PRNG.pobs, copula = copula)),
                                        mean(sobol_g(U.GMMN.QRNG.pobs, copula = copula)),
                                        if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                            mean(sobol_g(U.cop.QRNG, copula = copula)) else NA)
                                  } else rep(NA, 4)
                if(verbose) cat("=> Test function sobol_g() done\n")

                ## 3) Compute the exceedance probability over the (0.99,..,0.99) threshold
                ##    Note: Instead of Clayton, we use the survival Clayton copula here
                p <- 0.99
                trafo <- function(u) if(is(copula, "claytonCopula")) 1 - u else u
                raw[3,,nind,b] <- c(mean(exceedance(trafo(U.cop.PRNG),       q = p)),
                                    mean(exceedance(trafo(U.GMMN.PRNG.pobs), q = p)),
                                    mean(exceedance(trafo(U.GMMN.QRNG.pobs), q = p)),
                                    if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                        mean(exceedance(trafo(U.cop.QRNG), q = p)) else NA)
                if(verbose) cat("=> Test function exceedance() done\n")

                ## 4) Compute the p-level expected shortfall
                ##    Note: Instead of Clayton, we use the survival Clayton copula here
                raw[4,,nind,b] <- c(ES_np(qnorm(trafo(U.cop.PRNG)),       level = p),
                                    ES_np(qnorm(trafo(U.GMMN.PRNG.pobs)), level = p),
                                    ES_np(qnorm(trafo(U.GMMN.QRNG.pobs)), level = p),
                                    if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                        ES_np(qnorm(trafo(U.cop.QRNG)), level = p) else NA)
                if(verbose) cat("=> Test function ES_np() done\n")
            }
        }

        ## Compute errors, save and return
        res <- array(, dim = c(4, 4, nlen), dimnames = dmnms[1:3]) # result object
        res[1,,] <- apply(raw[1,,,], 1:2, mad) # apply mad() for fixed RNG, n combinations
        res[2,,] <- apply(raw[2,,,], 1:2, mad) # apply mad() for fixed RNG, n combinations
        res[3,,] <- apply(raw[3,,,], 1:2, sd)  # apply sd()  for fixed RNG, n combinations
        res[4,,] <- apply(raw[4,,,], 1:2, sd)  # apply sd()  for fixed RNG, n combinations
        saveRDS(res, file = file)
        res
    }
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
    if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file)
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
    if(doPDF) if(require(crop)) dev.off.crop(file) else dev.off(file)
}

##' @title Scatter Plots
##' @param u observations (from a PRNG, GMMN PRNG or GMMN QRNG)
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
##' @param CvM.testfun logical indicating whether the CvM statistics and
##'        the test functions are evaluated
##' @return nothing (computes results by side-effect)
##' @author Marius Hofert
##' @note Uses global variables to keep number of arguments small
main <- function(copula, name, model, CvM.testfun = TRUE)
{
    ## 1 Training ##############################################################

    ## Generate training data
    set.seed(271) # for reproducibility
    U <- rCopula(ntrn, copula = copula) # generate training dataset from a PRNG
    ## Train
    dim.in.out <- dim(copula) # dimension of the prior distribution fed into the GMMN
    NNname <- paste0("GMMN_QMC_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                     "_nbat_",nbat,"_nepo_",nepo,"_",name,".rda")
    GMMN <- train_once(dim = c(dim.in.out, dim.hid, dim.in.out), data = U,
                       batch.size = nbat, nepoch = nepo, file = NNname, package = "gnn")
    cat("=> Training done\n")

    ## 2 Contour/Rosenblatt plots or scatter plots #############################

    ## Setup and data generation
    bname <- paste0("dim_",dim.in.out,"_",name) # suffix
    seed <- 314
    set.seed(seed) # for reproducibility
    N01.prior.PRNG <- matrix(rnorm(ngen * dim.in.out), ncol = dim.in.out) # prior PRNs
    N01.prior.QRNG <- qnorm(sobol(ngen, d = dim.in.out, randomize = randomize, seed = seed)) # prior QRNs
    U.GMMN.PRNG <- pobs(predict(GMMN, x = N01.prior.PRNG)) # GMMN PRNs
    U.GMMN.QRNG <- pobs(predict(GMMN, x = N01.prior.QRNG)) # GMMN QRNs

    ## Contour, Rosenblatt and scatter plots
    if(dim.in.out == 2 && !grepl("MO", x = name)) { # rosenblatt() not available for copulas involving MO (MO itself or mixtures)
        contourplot3(copula, uPRNG = U.GMMN.PRNG, uQRNG = U.GMMN.QRNG,
                     file = paste0("GMMN_QMC_fig_contours_",bname,".pdf"))
        rosenplot(copula, u = U.GMMN.PRNG,
                  file = paste0("GMMN_QMC_fig_rosenblatt_",bname,".pdf"))
    }
    ## Scatter plots
    if(dim.in.out <= 3) { # for larger dimensions, one doesn't see much anyways
        lst <- list(PRNG = U[seq_len(ngen),], GMMN.PRNG = U.GMMN.PRNG, GMMN.QRNG = U.GMMN.QRNG)
        nms <- c("PRNG", "GMMN_PRNG", "GMMN_QRNG")
        for(i in seq_along(lst))
            scatterplot(lst[[i]], file = paste0("GMMN_QMC_fig_scatter_",bname,"_",nms[i],".pdf"))
    }
    cat("=> Contour, Rosenblatt and scatter plots done\n")

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
        CvMstat <- CvM(B.CvM, n = ngen, copula = copula, GMMN = GMMN,
                       randomize = randomize,
                       file = paste0("GMMN_QMC_comp_CvMstat_",bname,".rds"))
        cat("=> Computing Cramer-von Mises statistics done\n")

        ## Boxplots
        CvM_boxplot(CvMstat, dim = dim.in.out, model = model.,
                    file = paste0("GMMN_QMC_fig_CvMboxplot_",bname,".pdf"))

        ## 3.2 Test functions ##################################################

        ## Compute errors over B.conv replications; an (4, 4, length(ns))-array
        ## (<test function>, <RNG>, <sample size>)
        errTFs <- error_test_functions(B.conv, n = ns,
                                       copula = copula, GMMN = GMMN,
                                       randomize = randomize,
                                       file = paste0("GMMN_QMC_comp_testfun_",bname,".rds"))
        cat("=> Computing errors for test functions done\n")

        ## Plot convergence behavior
        convergence_plot(errTFs, dim = dim.in.out, model = model.,
                         filebname = paste0("GMMN_QMC_fig_convergence_",bname), B = B.conv)

    }
}

##' @title Results of the Appendix of the Paper
##' @param copula copula object
##' @param name character string (copula and taus together) for trained GMMNs,
##'        computed CvM statistics and test functions (.rds objects) as well as
##'        in corresponding boxplot and convergence plot figures
##' @param model call containing a model string used in boxplots and
##'        convergence plots; not used if CvM.testfun = FALSE
##' @return nothing (computes results by side-effect)
##' @author Marius Hofert
##' @note Uses global variables to keep number of arguments small
appendix <- function(copula, name, model)
{
    ## 1 Training ##############################################################

    ## Generate training data
    set.seed(271) # for reproducibility
    U <- rCopula(ntrn, copula = copula) # generate training dataset from a PRNG
    ## Train
    dim.in.out <- dim(copula) # dimension of the prior distribution fed into the GMMN
    NNname <- paste0("GMMN_QMC_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                     "_nbat_",nbat,"_nepo_",nepo,"_",name,".rda")
    GMMN <- train_once(dim = c(dim.in.out, dim.hid, dim.in.out), data = U,
                       batch.size = nbat, nepoch = nepo, file = NNname, package = "gnn")
    cat("=> Training done\n")

    ## 2 Expected shortfall test function ######################################

    bname <- paste0("dim_",dim.in.out,"_",name) # suffix
    file <- paste0("GMMN_QMC_comp_testfun_",bname,"_digital_shift.rds")
    res <- if (file.exists(file)) {
        readRDS(file)
    } else {
        ## Compute errors over B.conv replications; a (4, length(ns))-matrix
        ## (<RNG>, <sample size>)
        d <- dim(copula)
        n <- ns
        B <- B.conv
        nlen <- length(n)
        pb <- txtProgressBar(max = B, style = 3) # setup progress bar
        on.exit(close(pb)) # on exit, close progress bar
        dmnms <- list("RNG" = c("PRNG", "GMMN PRNG", "GMMN QRNG", "QRNG"),
                      "n" = as.character(n), "Replication" = 1:B)
        raw <- array(, dim = c(4, nlen, B), dimnames = dmnms) # intermediate object
        for(b in seq_len(B)) { # iterate over replications (just to have 'equidistant' progress bar)
            for(nind in seq_len(nlen)) { # iterate over sample sizes
                setTxtProgressBar(pb, b) # update progress bar
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
                ## If available in analytical form, draw from a real QRNG
                cCopula.inverse.avail <- is(copula, "normalCopula") || is(copula, "tCopula") ||
                    is(copula, "claytonCopula")
                if(cCopula.inverse.avail)
                    U.cop.QRNG <- cCopula(sob, copula = copula, inverse = TRUE)
                ## 1) Compute the p-level expected shortfall
                ##    Note: Instead of Clayton, we use the survival Clayton copula here
                p <- 0.99
                trafo <- function(u) if(is(copula, "claytonCopula")) 1 - u else u
                raw[,nind,b] <- c(ES_np(qnorm(trafo(U.cop.PRNG)),       level = p),
                                  ES_np(qnorm(trafo(U.GMMN.PRNG.pobs)), level = p),
                                  ES_np(qnorm(trafo(U.GMMN.QRNG.pobs)), level = p),
                                  if(cCopula.inverse.avail) # otherwise U.cop.QRNG doesn't exist
                                      ES_np(qnorm(trafo(U.cop.QRNG)), level = p) else NA)
            }
        }
        res. <- apply(raw, 1:2, sd) # apply sd() for fixed RNG, n combinations
        dimnames(res.) <- dmnms[1:2]
        saveRDS(res., file = file)
        res.
    }
    cat("\n=> Computing error for the test function done\n")

    ## Plot convergence behavior
    tau.str <- if(is(copula, "outer_nacopula")) {
                   th <- sort(unique(as.vector(nacPairthetas(copula))))
                   taus <- sapply(th, function(th.)
                       tau(archmCopula(copula@copula@name, param = th.)))
                   paste0("(",paste0(taus, collapse = ", "),")")
               } else as.character(tau(copula))
    model. <- substitute(m.*","~~tau==t., list(m. = model, t. = tau.str)) # model and taus
    convergence_plot(res, dim = dim.in.out, model = model.,
                     filebname = paste0("GMMN_QMC_fig_convergence_",bname,
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

### 2.1 Main part of the paper (timings are on a 13" MacBook Pro (2018) without training)

## Copulas from Section 1.1 above
system.time(main(t.cop.d2.tau1, name = paste0("t",nu,"_tau_",taus[1]), # ~= 2s
                 model = quote(italic(t)[4]), CvM.testfun = FALSE))
system.time(main(t.cop.d2.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 16min
                 model = quote(italic(t)[4])))
system.time(main(t.cop.d2.tau3, name = paste0("t",nu,"_tau_",taus[3]), # ~= 2s
                 model = quote(italic(t)[4]), CvM.testfun = FALSE))
system.time(main(C.cop.d2.tau1, name = paste0("C","_tau_",taus[1]), # ~= 2s
                 model = quote(Clayton), CvM.testfun = FALSE))
system.time(main(C.cop.d2.tau2, name = paste0("C","_tau_",taus[2]), # ~= 7min
                 model = quote(Clayton)))
system.time(main(C.cop.d2.tau3, name = paste0("C","_tau_",taus[3]), # ~= 2s
                 model = quote(Clayton), CvM.testfun = FALSE))
system.time(main(G.cop.d2.tau1, name = paste0("G","_tau_",taus[1]), # ~= 2s
                 model = quote(Gumbel), CvM.testfun = FALSE))
system.time(main(G.cop.d2.tau2, name = paste0("G","_tau_",taus[2]), # ~= 12min
                 model = quote(Gumbel)))
system.time(main(G.cop.d2.tau3, name = paste0("G","_tau_",taus[3]), # ~= 2s
                 model = quote(Gumbel), CvM.testfun = FALSE))
system.time(main(MO.cop.d2, name = paste0("MO_",paste0(alpha,collapse = "_")), # ~= 2s
                 model = quote(MO), CvM.testfun = FALSE)) # argument 'model' actually not used here
system.time(main(mix.cop.C.t90, name = "eqmix_C_tau_0.5_rot90_t4_tau_0.5", # ~= 2s
                 model = quote(Clayton-italic(t)[4](90)), CvM.testfun = FALSE)) # argument 'model' actually not used here
system.time(main(mix.cop.G.t90,  name = "eqmix_G_tau_0.5_rot90_t4_tau_0.5", # ~= 2s
                 model = quote(Gumbel-italic(t)[4](90)), CvM.testfun = FALSE)) # argument 'model' actually not used here
system.time(main(mix.cop.MO.t90, # ~= 2s
                 name = paste0("eqmix_MO_",paste0(alpha,collapse = "_"),"_rot90_t4_tau_0.5"),
                 model = quote(MO-italic(t)[4](90)), CvM.testfun = FALSE)) # argument 'model' actually not used here

## Copulas from Section 1.2 above
system.time(main(NC.d21, name = paste0("NC21_tau_",paste0(taus[1:2], collapse = "_")), # ~= 5min
                 model = quote("(2,1)-nested Clayton")))
system.time(main(NG.d21, name = paste0("NG21_tau_",paste0(taus[1:2], collapse = "_")), # ~= 5min
                 model = quote("(2,1)-nested Gumbel")))

## Copulas from Section 1.3 above
system.time(main(t.cop.d5.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 60min
                 model = quote(italic(t)[4])))
system.time(main(C.cop.d5.tau2, name = paste0("C","_tau_",taus[2]), # ~= 9min
                 model = quote(Clayton)))
system.time(main(G.cop.d5.tau2, name = paste0("G","_tau_",taus[2]), # ~= 32min
                 model = quote(Gumbel)))
system.time(main(NC.d23, name = paste0("NC23_tau_",paste0(taus, collapse = "_")), # ~= 5min
                 model = quote("(2,3)-nested Clayton")))
system.time(main(NG.d23, name = paste0("NG23_tau_",paste0(taus, collapse = "_")), # ~= 5min
                 model = quote("(2,3)-nested Gumbel")))

## Copulas from Section 1.4 above
system.time(main(t.cop.d10.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 4.4h
                 model = quote(italic(t)[4])))
system.time(main(C.cop.d10.tau2, name = paste0("C","_tau_",taus[2]), # ~= 13min
                 model = quote(Clayton)))
system.time(main(G.cop.d10.tau2, name = paste0("G","_tau_",taus[2]), # ~= 1.1h
                 model = quote(Gumbel)))
system.time(main(NC.d55, name = paste0("NC55_tau_",paste0(taus, collapse = "_")), # ~= 13min
                 model = quote("(5,5)-nested Clayton")))
system.time(main(NG.d55, name = paste0("NG55_tau_",paste0(taus, collapse = "_")), # ~= 12min
                 model = quote("(5,5)-nested Gumbel")))


### 2.2 Appendix (done 'manually' here for randomize = "digital.shift")

## Note: No .rds will be written in this case (just the plots generated directly)

## Row 1
system.time(appendix(t.cop.d2.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 5min
                     model = quote(italic(t)[4])))
system.time(appendix(t.cop.d5.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 11min
                     model = quote(italic(t)[4])))
system.time(appendix(t.cop.d10.tau2, name = paste0("t",nu,"_tau_",taus[2]), # ~= 20min
                     model = quote(italic(t)[4])))
## Row 2
system.time(appendix(C.cop.d2.tau2, name = paste0("C","_tau_",taus[2]), # ~= 4min
                     model = quote(Clayton)))
system.time(appendix(C.cop.d5.tau2, name = paste0("C","_tau_",taus[2]), # ~= 5min
                     model = quote(Clayton)))
system.time(appendix(C.cop.d10.tau2, name = paste0("C","_tau_",taus[2]), # ~= 7min
                     model = quote(Clayton)))
## Row 3
system.time(appendix(G.cop.d2.tau2, name = paste0("G","_tau_",taus[2]), # ~= 4min
                     model = quote(Gumbel)))
system.time(appendix(G.cop.d5.tau2, name = paste0("G","_tau_",taus[2]), # ~= 5min
                     model = quote(Gumbel)))
system.time(appendix(G.cop.d10.tau2, name = paste0("G","_tau_",taus[2]), # ~= 7min
                     model = quote(Gumbel)))
## Row 4
system.time(appendix(NG.d21, name = paste0("NG21_tau_",paste0(taus[1:2], collapse = "_")), # ~= 5min
                     model = quote("(2,1)-nested Gumbel")))
system.time(appendix(NG.d23, name = paste0("NG23_tau_",paste0(taus, collapse = "_")), # ~= 6min
                     model = quote("(2,3)-nested Gumbel")))
system.time(appendix(NG.d55, name = paste0("NG55_tau_",paste0(taus, collapse = "_")), # ~= 7min
                     model = quote("(5,5)-nested Gumbel")))
