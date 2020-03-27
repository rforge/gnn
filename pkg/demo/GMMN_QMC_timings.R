## By Marius Hofert and Avinash Prasad

## Measuring run time for certain examples in Hofert, Prasad, Zhu ("Quasi-random
## sampling for multivariate distributions via generative neural networks").
## Training times are obtained from running this script on an NVIDIA Tesla P100
## GPU (about 3h), run times of the various methods are obtained from running
## this script locally on a 13" MacBook Pro (2018) in about 30min as the GPU's
## run time measurements weren't reliable.


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # note: access of functions via '::' fails for this package
## => would allow to set the seed with use_session_with_seed(271), but then no GPU or CPU parallelism
if(grepl("gra", Sys.info()[["nodename"]])) {
    tf_version() # dummy call to activate connection to TensorFlow (any first call will fail on the cluster; here: NULL)
    use_virtualenv(Sys.getenv('VIRTUAL_ENV')) # tensorflow command to access the activated Python environment
}
if(packageVersion("qrng") < "0.0.7")
    stop('Consider updating via install.packages("qrng", repos = "http://R-Forge.R-project.org")')
library(qrng) # for sobol()
if(packageVersion("copula") < "0.999.19")
    stop('Consider updating via install.packages("copula", repos = "http://R-Forge.R-project.org")')
library(copula) # for the considered copulas
library(gnn) # for the used GMMN models
library(qrmtools) # for catch()
library(simsalapar) # for toLatex()

## Global training parameters
dim.hid <- 300L # dimension of the (single) hidden layer
ntrn <- 60000L # training dataset size (number of pseudo-random numbers from the copula)
nbat <- 5000L # batch size for training (number of samples per stochastic gradient step)
nepo <- 300L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
stopifnot(dim.hid >= 1, ntrn >= 1, 1 <= nbat, nbat <= ntrn, nepo >= 1)

## Number of samples to be generated (for those copulas with cCopula() analytically
## available and for all other copulas, respectively)
ngen <- c(1e5, 1e3) # or 1e6 (about 30min), 1e7 (about 3h)


### 0 Auxiliary functions ######################################################

##' @title Measuring Training Time of a GMMN
##' @param copula copula object
##' @param cstrng character string (copula and taus together) for the trained GMMN
##' @return list containing the trained GMMN and elapsed training time
training_time <- function(copula, cstrng)
{
    U <- rCopula(ntrn, copula = copula) # pseudo-random training data
    dim.in.out <- dim(copula) # = dimension of the prior distribution fed into the GMMN
    file <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                   "_nbat_",nbat,"_nepo_",nepo,"_",cstrng,".rda")
    name <- rm_ext(basename(file))
    if(exists_rda(file, names = name)) {
        read.gnn <- read_rda(file, names = name)
        GMMN <- to_callable(read.gnn)
        train.time <- NA
    } else {
        GNN <- GMMN_model(c(dim.in.out, dim.hid, dim.in.out)) # GMMN model
        train.time <- system.time(GMMN <- train(GNN, data = U, # train and measure elapsed time
                                                batch.size = nbat, nepoch = nepo))["elapsed"] / 60 # in min
        save_rda(to_savable(GMMN), file = file, names = name) # save
        ## Note: We use train() instead of train_once() to capture only training time.
        ##       Also, if a saved GMMN exists in the current working directory
        ##       train_once() will lead to misleading training times.
    }
    list(GMMN = GMMN, train.time = train.time) # return the trained GMMN along with its training time
}

##' @title Measuring Run Times for Copula PRS, Copula QRS, GMMN PRS, GMMN QRS
##' @param copula copula object
##' @param GMMN GMMN trained on pseudo-random samples from 'copula'
##' @param seed seed to be used for reproducibility
##' @return list of average run times in s for each of the four sampling methods
run_times <- function(copula, GMMN, seed = 271)
{
    ## Run time measurement per 1M random variates
    timer <- function(expr) system.time(expr)[["elapsed"]] # auxiliary function

    ## Copula PRS (here we can easily take replicates as there is no problem with the seed)
    set.seed(271)
    rt.cop.PRS <- mean(replicate(100, expr = timer(rCopula(ngen[1], copula = copula)))) # mean time in sec over 100 replications

    ## Numerically robust QRS
    rQRS <- function(n, copula, seed) {
        res <- catch(cCopula(sobol(n, d = dim(copula), randomize = "Owen", seed = seed),
                             copula = copula, inverse = TRUE))
        if(is.null(res$error)) res$value else NULL # NULL if there was an error
        ## Note: We use NULL instead of NA as that test works for matrices and NULL
    }

    ## Copula QRS
    ## Note:
    ## - Depending on the copulas, we choose a smaller sample size and then scale up
    ##   run time afterwards.
    ## - We also need to check whether there were numerical problems and then set
    ##   the run time to NA
    cond.cop.analytical <- is(copula, "normalCopula") || is(copula, "tCopula") || is(copula, "claytonCopula")
    ngen. <- if(cond.cop.analytical) ngen[1] else ngen[2]
    rt.cop.QRS <- timer(res.cop.QRS <- rQRS(ngen., copula = copula, seed = seed))
    if(is.null(res.cop.QRS)) {
        rt.cop.QRS <- NA # if there was an error when generating ngen.-many realizations (e.g. for NG)
    } else { # no error, then scale again if necessary
        if(!cond.cop.analytical) rt.cop.QRS <- rt.cop.QRS * ngen[1]/ngen[2] # scale up run time (e.g. for G)
    }
    ## Expected behavior: scaled run time for Gumbel, NA for nested Gumbel

    ## GMMN PRS (no need for replicates)
    d <- dim(copula)
    set.seed(271)
    rt.GMMN.PRS <- timer(pobs(predict(GMMN, x = matrix(rnorm(ngen[1] * d), ncol = d))))

    ## GMMN QRS (no need for replicates; would also make seed passing more difficult)
    rt.GMMN.QRS <- timer(pobs(predict(GMMN, x = qnorm(sobol(ngen[1], d = d, randomize = "Owen", seed = seed)))))

    ## Return
    c("Copula PRS" = rt.cop.PRS,  "Copula QRS" = rt.cop.QRS,
      "GMMN PRS"   = rt.GMMN.PRS, "GMMN QRS"   = rt.GMMN.QRS)
}

##' @title Run Times for GMMN Training and Copula PRS, Copula QRS, GMMN PRS, GMMN QRS
##' @param copula see ?training_time or ?sampling_time
##' @param cstrng see ?training_time
##' @param file character string specifying the file name used to save results
##' @return list containing GMMN training time and sub-list containing (average)
##'         run time for each of the four sampling methods
timings <- function(copula, cstrng, file)
{
    if(file.exists(file)) {
        res <- readRDS(file)
    } else {
        training <- training_time(copula, cstrng = cstrng) # GMMN training
        res <- list(training.time = training$train.time, # measure run times
                    run.times = run_times(copula, GMMN = training$GMMN[["model"]]))
        saveRDS(res, file = file) # save
    }
    res
}

##' @title Load and Extract the Model, Parameters and Dimensions
##' @param x string specifying the GMMN
##' @return 3-list containing the model, parameters and dimensions of all layers
##' @author Marius Hofert
get_model_param_dim <- function(x)
{
    load(paste0(x,".rda"))
    gnn <- get(x)
    model <- to_callable(gnn)[["model"]] # get trained model
    param <- get_weights(model)
    names(param) <- paste0(c("W", "b"), rep(seq_len(length(gnn[["dim"]]) -1), each = 2))
    list(model = model, param = param, dim = gnn[["dim"]])
}

##' @title Compute a Linear Transformation
##' @param x input
##' @param A matrix
##' @param b vector
##' @return x A + b
##' @author Marius Hofert
trafo <- function(x, A, b) x %*% A + rep(b, each = nrow(x))

##' @title R Implementation of predict()
##' @param x evaluation points
##' @param params GMMN parameters
##' @return GMMN output
##' @author Marius Hofert
predictR <- function(x, param)
    plogis(trafo(pmax(trafo(x, A = param$W1, b = param$b1), 0), A = param$W2, b = param$b2))


### 1 Copula objects used ######################################################

taus <- c(0.25, 0.5, 0.75) # Kendall's taus considered


### 1.1 d = 2 ##################################################################

d.2 <- 2 # copula dimension

## t copula
nu <- 4 # degrees of freedom of the t copula
th.t <- iTau(tCopula(), tau = taus[2]) # parameter
t.cop.d2 <- tCopula(th.t, dim = d.2, df = nu)

## Clayton copula
th.C <- iTau(claytonCopula(), tau = taus[2])
C.cop.d2 <- claytonCopula(th.C, dim = d.2)

## Gumbel copula
th.G <- iTau(gumbelCopula(), tau = taus[2])
G.cop.d2 <- gumbelCopula(th.G, dim = d.2)


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
ds.3 <- c(2, 1) # sector dimensions
th.Gs <- iTau(gumbelCopula(), tau = taus)
NG.d21 <- onacopulaL("Gumbel",  nacList = nacList(ds.3, th = th.Gs)) # nested Gumbel


### 1.3 d = 5 ##################################################################

d.5 <- 5 # copula dimension

## t copula
t.cop.d5 <- tCopula(th.t, dim = d.5, df = nu)

## Clayton copula
C.cop.d5 <- claytonCopula(th.C, dim = d.5)

## Gumbel copula
G.cop.d5 <- gumbelCopula(th.G, dim = d.5)

## Nested copulas
ds.5 <- c(2, 3) # sector dimensions
NG.d23 <- onacopulaL("Gumbel",  nacList = nacList(ds.5, th = th.Gs)) # nested Gumbel


### 1.4 d = 10 #################################################################

d.10 <- 10 # copula dimension

## t copula
t.cop.d10 <- tCopula(th.t, dim = d.10, df = nu)

## Clayton copula
C.cop.d10 <- claytonCopula(th.C, dim = d.10)

## Gumbel copula
G.cop.d10 <- gumbelCopula(th.G, dim = d.10)

## Nested copulas
ds.10 <- c(5, 5) # sector dimensions
NG.d55 <- onacopulaL("Gumbel",  nacList = nacList(ds.10, th = th.Gs)) # nested Gumbel


### 2 GMMN Training and Run Time Measurement ###################################

## d = 2 dimensional copulas
res.t.d2 <- timings(t.cop.d2, cstrng = paste0("t",nu,"_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.2,"_t",nu,"_tau_",taus[2],".rds"))
res.C.d2 <- timings(C.cop.d2, cstrng = paste0("C","_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.2,"_C","_tau_",taus[2],".rds"))
res.G.d2 <- timings(G.cop.d2, cstrng = paste0("G","_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.2,"_G","_tau_",taus[2],".rds"))

## d = 3 dimensional copulas
res.NG.d21 <- timings(NG.d21, cstrng = paste0("NG21_tau_",paste0(taus[1:2], collapse = "_")),
                      file = paste0("timing","_dim_",sum(ds.3),"_NG21","_tau_",
                                    paste0(taus[1:2], collapse = "_",".rds")))

## d = 5 dimensional copulas
res.t.d5 <- timings(t.cop.d5, cstrng = paste0("t",nu,"_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.5,"_t",nu,"_tau_",taus[2],".rds"))
res.C.d5 <- timings(C.cop.d5, cstrng = paste0("C","_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.5,"_C","_tau_",taus[2],".rds"))
res.G.d5 <- timings(G.cop.d5, cstrng = paste0("G","_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.5,"_G","_tau_",taus[2],".rds"))
res.NG.d23 <- timings(NG.d23, cstrng = paste0("NG23_tau_",paste0(taus, collapse = "_")),
                      file = paste0("timing","_dim_",sum(ds.5),"_NG23","_tau_",
                                    paste0(taus, collapse = "_",".rds")))

## d = 10 dimensional copulas
res.t.d10 <- timings(t.cop.d10, cstrng = paste0("t",nu,"_tau_",taus[2]),
                     file = paste0("timing","_dim_",d.10,"_t",nu,"_tau_",taus[2],".rds"))
res.C.d10 <- timings(C.cop.d10, cstrng = paste0("C","_tau_",taus[2]),
                     file = paste0("timing","_dim_",d.10,"_C","_tau_",taus[2],".rds"))
res.G.d10 <- timings(G.cop.d10, cstrng = paste0("G","_tau_",taus[2]),
                     file = paste0("timing","_dim_",d.10,"_G","_tau_",taus[2],".rds"))
res.NG.d55 <- timings(NG.d55, cstrng = paste0("NG55_tau_",paste0(taus, collapse = "_")),
                      file = paste0("timing","_dim_",sum(ds.10),"_NG55","_tau_",
                                    paste0(taus, collapse = "_",".rds")))


### 3 Results ##################################################################

cops <- c(paste0("t",nu), "C", "G", "NG") # copulas
dms <- c("d = 2, 3", "d = 5", "d = 10") # dimensions
meths <- c("Copula PRS", "Copula QRS", "GMMN PRS", "GMMN QRS") # methods


### 3.1 Training times #########################################################

## Collect results in a matrix
res.training <- matrix(noquote(sprintf("%.2f", c(
    res.t.d2$training.time,   res.t.d5$training.time,   res.t.d10$training.time,
    res.C.d2$training.time,   res.C.d5$training.time,   res.C.d10$training.time,
    res.G.d2$training.time,   res.G.d5$training.time,   res.G.d10$training.time,
    res.NG.d21$training.time, res.NG.d23$training.time, res.NG.d55$training.time))),
    ncol = 3, byrow = TRUE, dimnames = list("Copula" = cops, "Dimension" = dms))

## Convert to table
toLatex(ftable(res.training))


### 3.2 Run times when sampling ################################################

## Collect results in a matrix
res.sampling <- matrix(noquote(sprintf("%.4f", c(
    ## d = 2 and d = 3 dimensional copulas
    res.t.d2$run.times,
    res.C.d2$run.times,
    res.G.d2$run.times,
    res.NG.d21$run.times,
    ## d = 5 dimensional copulas
    res.t.d5$run.times,
    res.C.d5$run.times,
    res.G.d5$run.times,
    res.NG.d23$run.times,
    ## d = 10 dimensional copulas
    res.t.d10$run.times,
    res.C.d10$run.times,
    res.G.d10$run.times,
    res.NG.d55$run.times))),
    ncol = 4, byrow = TRUE, dimnames = list("Copula" = rep(cops, 3), "Method" = meths))

## Convert to table
toLatex(ftable(res.sampling))


### 4 Run time comparison of R implementation in comparison to TensorFlow ######

## GMMNs considered
GMMNs <- c("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5",
           "GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5",
           "GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5")


### 4.1 Do we match the output of TensorFlow ###################################

mod <- get_model_param_dim(GMMNs[1])
d <- mod$dim[1]
n <- 1e5
set.seed(271)
U. <- matrix(rnorm(n * d), ncol = d)
system.time(U.TF <- predict(mod$model, x = U.))
system.time(U.R  <- predictR(U., param = mod$param))
stopifnot(all.equal(U.TF, U.R, tolerance = 1e-7))


### 4.2 Run time as a function of n for several GMMNs ##########################

## Compute run times as a function of n
ngen. <- 10^seq(1, 6, by = 0.5) # sample sizes considered
d <- c(2, 5, 10) # dimensions considered
set.seed(271); U. <- matrix(rnorm(max(ngen.) * max(d)), ncol = max(d)) # generate for the max. sample size and dimension
res <- matrix(, nrow = length(ngen.), ncol = length(d)) # (length(n), length(d))-matrix
pb <- txtProgressBar(max = length(d) * length(ngen.), style = 3)
for(j in seq_along(d)) {
    mod <- get_model_param_dim(GMMNs[j])
    for(i in seq_along(ngen.)) {
        U.. <- U.[1:ngen.[i], 1:d[j], drop = FALSE]
        res[i, j] <-
            mean(replicate(10, expr = system.time(predictR(U.., param = mod$param))[["elapsed"]] /
                                   system.time(predict(mod$model, x = U..))[["elapsed"]]))
        setTxtProgressBar(pb, length(ngen.) * (j-1) + i)
    }
}
close(pb)

## Plot
file <- "fig_elapsed_time_R_over_TensorFlow.pdf"
pdf(file, bg = "transparent", width = 7, height = 7)
opar <- par(pty = "s")
plot(ngen., res[,1], type = "l", log = "x", ylim = range(res), xaxt = "n",
     xlab = expression(n[gen]),
     ylab = "Elapsed time of R implementation / TensorFlow implementation")
labels <- sapply(1:length(ngen.), function(i) if(i %% 2 == 1) as.expression(bquote(10^.((i+1)/2))) else NA)
axis(1, at = ngen., labels = labels)
lines(ngen., res[,2], lty = 2)
lines(ngen., res[,3], lty = 3)
abline(h = 1, lty = 4)
legend("bottomright", bty = "n", col = 1, lty = 1:3, legend = paste("d =", d))
mtext(substitute(italic(t)[nu.]~"copula with Kendall's tau"~tau==tau.,
                 list(nu. = nu, tau. = taus[2])),
      side = 4, line = 0.5, adj = 0)
par(opar)
if(require(crop)) dev.off.crop(file) else dev.off(file)
