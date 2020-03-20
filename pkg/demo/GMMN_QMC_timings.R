## By Marius Hofert and Avinash Prasad

## Measuring run time for certain examples in Hofert, Prasad, Zhu ("Quasi-random
## sampling for multivariate distributions via generative neural networks"). The
## NNs were trained on an NVIDIA Tesla P100 GPU.


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # note: access of functions via '::' fails for this package
## => would allow to set the seed with use_session_with_seed(271), but then no GPU or CPU parallelism
tf_version() # dummy call to activate connection to TensorFlow (any first call will fail on the cluster; here: NULL)
use_virtualenv(Sys.getenv('VIRTUAL_ENV')) # tensorflow command to access the activated Python environment
if(packageVersion("qrng") < "0.0.7")
    stop('Consider updating via install.packages("qrng", repos = "http://R-Forge.R-project.org")')
library(qrng) # for sobol()
if(packageVersion("copula") < "0.999.19")
    stop('Consider updating via install.packages("copula", repos = "http://R-Forge.R-project.org")')
library(copula) # for the considered copulas
library(gnn) # for the used GMMN models
library(microbenchmark) # for measuring run time
library(qrmtools) # for catch()
library(simsalapar) # for toLatex()

## Global training parameters
package <- NULL # use "gnn" to utilize pre-trained NNs from gnn
dim.hid <- 300L # dimension of the (single) hidden layer
ntrn <- 60000L # training dataset size (number of pseudo-random numbers from the copula)
nbat <- 5000L # batch size for training (number of samples per stochastic gradient step)
nepo <- 300L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
stopifnot(dim.hid >= 1, ntrn >= 1, 1 <= nbat, nbat <= ntrn, nepo >= 1)

## Other global parameters
ngen <- 1e6 # sample size of the generated data


### 0 Auxiliary functions ######################################################

##' @title Measuring Training Time of a GMMN
##' @param copula copula object
##' @param name character string (copula and taus together) for the trained GMMN
##' @return list containing the trained GMMN and elapsed training time
training_time <- function(copula, name)
{
    U <- rCopula(ntrn, copula = copula) # pseudo-random training data
    dim.in.out <- dim(copula) # = dimension of the prior distribution fed into the GMMN
    NNname <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                     "_nbat_",nbat,"_nepo_",nepo,"_",name,".rda")
    GNN <- GMMN_model(c(dim.in.out, dim.hid, dim.in.out)) # GMMN model
    train.time <- system.time(GMMN <- train(GNN, data = U, # train and measure elapsed time
                                            batch.size = nbat, nepoch = nepo))["elapsed"] / 60 # in min
    save_rda(to_savable(GMMN), file = NNname, names = rm_ext(basename(NNname))) # save
    ## Note: We use train() instead of train_once() to capture only training time.
    ##       Also, if a saved GMMN exists in the current working directory or
    ##       if package = "gnn", train_once() will lead to misleading training times.
    list(GMMN = GMMN, train.time = train.time) # return the trained GMMN along with its training time
}

##' @title Measuring Run Times for Copula PRS, Copula QRS, GMMN PRS, GMMN QRS
##' @param copula copula object
##' @param GMMN GMMN trained on pseudo-random samples from 'copula'
##' @param times number of replications
##' @param unit the unit in which run time is measured
##' @return list of average run times in s for each of the four sampling methods
run_times <- function(copula, GMMN, times = 25, unit = "s", seed = 271)
{
    ## Set seed (we do that here to use common random variates)
    set.seed(seed)

    ## QRNG
    scale <- if(is(copula, "gumbelCopula")) 1000 else 1 # for scaling in case of Gumbel
    rQRS <- function(n, copula, d, seed) {
        ## catch errors
        res <- catch(cCopula(sobol(n / scale, d = d, randomize = "Owen", seed = seed),
                             copula = copula, inverse = TRUE))
        if(is.null(res$error)) res$value else NA # NA if not available or if there was an error
    }

    ## Run time measurement for Copula PRS, Copula QRS, GMMN PRS, GMMN QRS
    d <- dim(copula) # copula dimension
    rt <- summary(microbenchmark(
        rCopula(ngen, copula = copula), # Copula PRS
        rQRS(ngen, copula = copula, d = d, seed = seed), # Copula QRS
        predict(GMMN, x = matrix(rnorm(ngen * d), ncol = d)), # GMMN PRS
        predict(GMMN, x = qnorm(sobol(ngen, d = d, randomize = "Owen", seed = seed))), # GMMN QRS
        times = times, unit = unit))$mean
    names(rt) <- c("Copula PRS", "Copula QRS", "GMMN PRS", "GMMN QRS")

    ## Adjust run time for Gumbel; see (*)
    if(is(copula, "gumbelCopula")) rt[2] <- scale * rt[2]

    ## Return
    rt
}

##' @title Run Times for GMMN Training and Copula PRS, Copula QRS, GMMN PRS, GMMN QRS
##' @param copula see ?training_time or ?sampling_time
##' @param name see ?training_time
##' @param file character string specifying the file name used to save results
##' @return list containing GMMN training time and sub-list containing (average)
##'         run time for each of the four sampling methods
timings <- function(copula, name, file)
{
    if(file.exists(file)) {
        res <- readRDS(file)
    } else {
        training <- training_time(copula, name = name) # GMMN training
        res <- list(training.time = training$train.time, # measure run times
                    run.times = run_times(copula, GMMN = training$GMMN[["model"]]))
        saveRDS(res, file = file) # save
    }
    res
}


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
res.t.d2 <- timings(t.cop.d2, name = paste0("t",nu,"_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.2,"_t",nu,"_tau_",taus[2],".rds"))
res.C.d2 <- timings(C.cop.d2, name = paste0("C","_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.2,"_C","_tau_",taus[2],".rds"))
res.G.d2 <- timings(G.cop.d2, name = paste0("G","_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.2,"_G","_tau_",taus[2],".rds"))

## d = 3 dimensional copulas
res.NG.d21 <- timings(NG.d21, name = paste0("NG21_tau_",paste0(taus[1:2], collapse = "_")),
                      file = paste0("timing","_dim_",sum(ds.3),"_NG21","_tau_",
                                    paste0(taus[1:2], collapse = "_",".rds")))

## d = 5 dimensional copulas
res.t.d5 <- timings(t.cop.d5, name = paste0("t",nu,"_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.5,"_t",nu,"_tau_",taus[2],".rds"))
res.C.d5 <- timings(C.cop.d5, name = paste0("C","_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.5,"_C","_tau_",taus[2],".rds"))
res.G.d5 <- timings(G.cop.d5, name = paste0("G","_tau_",taus[2]),
                    file = paste0("timing","_dim_",d.5,"_G","_tau_",taus[2],".rds"))
res.NG.d23 <- timings(NG.d23, name = paste0("NG23_tau_",paste0(taus, collapse = "_")),
                      file = paste0("timing","_dim_",sum(ds.5),"_NG23","_tau_",
                                    paste0(taus, collapse = "_",".rds")))

## d = 10 dimensional copulas
res.t.d10 <- timings(t.cop.d10, name = paste0("t",nu,"_tau_",taus[2]),
                     file = paste0("timing","_dim_",d.10,"_t",nu,"_tau_",taus[2],".rds"))
res.C.d10 <- timings(C.cop.d10, name = paste0("C","_tau_",taus[2]),
                     file = paste0("timing","_dim_",d.10,"_C","_tau_",taus[2],".rds"))
res.G.d10 <- timings(G.cop.d10, name = paste0("G","_tau_",taus[2]),
                     file = paste0("timing","_dim_",d.10,"_G","_tau_",taus[2],".rds"))
res.NG.d55 <- timings(NG.d55, name = paste0("NG55_tau_",paste0(taus, collapse = "_")),
                      file = paste0("timing","_dim_",sum(ds.10),"_NG55","_tau_",
                                    paste0(taus, collapse = "_",".rds")))


### 3 Results ##################################################################

cops <- c(paste0("t",nu), "C", "G", "NG") # copulas
dms <- c("d = 2, 3", "d = 5", "d = 10") # dimensions
meths <- c("Copula PRS", "Copula QRS", "GMMN PRS", "GMMN QRS") # methods


### 3.1 Training times #########################################################

## Collect results in a matrix
res.training <- matrix(c(
    res.t.d2$training.time,   res.t.d5$training.time,   res.t.d10$training.time,
    res.C.d2$training.time,   res.C.d5$training.time,   res.C.d10$training.time,
    res.G.d2$training.time,   res.G.d5$training.time,   res.G.d10$training.time,
    res.NG.d21$training.time, res.NG.d23$training.time, res.NG.d55$training.time),
    ncol = 4, byrow = TRUE, dimnames = list("Copula" = cops, "Dimension" = dms))

## Convert to table
toLatex(ftable(res.training))


### 3.2 Run times when sampling ################################################

## Collect results in a matrix
res.sampling <- matrix(c(
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
    res.NG.d55$run.times), ncol = 4, byrow = TRUE,
    dimnames = list("Copula" = cops, "Method" = meths))

## Convert to table
toLatex(ftable(res.sampling))
