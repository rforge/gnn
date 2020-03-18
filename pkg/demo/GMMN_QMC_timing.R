## By Marius Hofert and Avinash Prasad

## Measuring run time for certain examples in Hofert, Prasad, Zhu ("Quasi-random sampling
## for multivariate distributions via generative neural networks"). The NNs were trained on
## an NVIDIA Tesla P100 GPU.


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

## Global training parameters
package <- NULL # use "gnn" to utilize pre-trained NNs from gnn
dim.hid <- 300L # dimension of the (single) hidden layer
ntrn <- 60000L # training dataset size (number of pseudo-random numbers from the copula)
nbat <- 5000L # batch size for training (number of samples per stochastic gradient step)
nepo <- 300L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
stopifnot(dim.hid >= 1, ntrn >= 1, 1 <= nbat, nbat <= ntrn, nepo >= 1)

## Other global parameters
ngen <- 1e4 # sample size of the generated data


### 0 Auxiliary functions ######################################################

##' @title Measuring Training Time of a GMMN
##' @param copula copula object
##' @param name character string (copula and taus together) for the trained GMMN
##' @return list containing the trained GMMN and elapsed training time
training_time <- function(copula, name)
{
    U <- rCopula(ntrn, copula = copula) # generate training data from a PRNG
    dim.in.out <- dim(copula) # = dimension of the prior distribution fed into the GMMN
    NNname <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                     "_nbat_",nbat,"_nepo_",nepo,"_",name,".rda")
    GNN <- GMMN_model(c(dim.in.out, dim.hid, dim.in.out)) # GMMN model
    train.time <- human_time(GMMN <- train(GNN, data = U, # train and measure elapsed time
                                           batch.size = nbat, nepoch = nepo))["elapsed"]
    save_rda(to_savable(GMMN), file = NNname, names = rm_ext(basename(NNname))) # save
    ## Note: We use train() instead of train_once() to capture only training time.
    ##       Also, if a saved GMMN exists in the current working directory or
    ##       if package = "gnn", train_once() will lead to misleading training times.
    list(GMMN = GMMN, train.time = train.time) # return the trained GMMN along with its training time
}

##' @title Measuring Run Times for PRNG, GMMN PRNG, QRNG, GMMN QRNG
##' @param copula copula object
##' @param GMMN GMMN trained on pseudo-random samples from 'copula'
##' @param times number of replications
##' @param unit the unit in which run time is measured
##' @return list of average run times in s for each of the four sampling methods
run_times <- function(copula, GMMN, times = 100, unit = "s")
{
    d <- dim(copula) # copula dimension
    cCopula.inv.avail <- is(copula, "normalCopula") || is(copula, "tCopula") ||
        is(copula, "claytonCopula") # check whether inverse Rosenblatt transform is available
    rt <- summary(microbenchmark(
        rCopula(ngen, copula = copula), # PRNG
        predict(GMMN, x = matrix(rnorm(ngen * d), ncol = d)), # GMMN PRNG
        if(cCopula.inv.avail) cCopula(sobol(ngen, d = d, randomize = "Owen", seed = 271),
                                      copula = copula, inverse = TRUE) else NA, # QRNG
        predict(GMMN, x = qnorm(sobol(ngen, d = d, randomize = "Owen", seed = 271))), # GMMN QRNG
        times = times, unit = unit))$mean
    names(rt) <- c("PRNG", "GMMN.PRNG", "QRNG", "GMMN.QRNG")
    rt
}

##' @title Run Times for GMMN Training and PRNG, GMMN PRNG, QRNG, GMMN QRNG Sampling
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
        set.seed(271) # set seed here to use common random variates
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


### 1.4 d = 10 ##################################################################

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

## Copulas from Section 1.1 above
timings(copula=t.cop.d2,name = paste0("t",nu,"_tau_",taus[2]),
       file=paste0("timing","_dim_",d,"_t",nu,"_tau_",taus[2],".rds"))
timings(copula=C.cop.d2,name= paste0("C","_tau_",taus[2]),
       file=paste0("timing","_dim_",d,"_C","_tau_",taus[2],".rds"))
timings(copula=G.cop.d2,name= paste0("G","_tau_",taus[2]),
       file=paste0("timing","_dim_",d,"_G","_tau_",taus[2],".rds"))

## Copulas from Section 1.2 above
timings(copula=NG.d21,name=paste0("NG21_tau_",paste0(taus[1:2], collapse = "_")),
       file=paste0("timing","_dim_",sum(ds),"_NG21","_tau_",
                   paste0(taus[1:2], collapse = "_",".rds")))

## Copulas from Section 1.3 above
timings(copula=t.cop.d5,name = paste0("t",nu,"_tau_",taus[2]),
       file=paste0("timing","_dim_",d.5,"_t",nu,"_tau_",taus[2],".rds"))
timings(copula=C.cop.d5,name= paste0("C","_tau_",taus[2]),
       file=paste0("timing","_dim_",d.5,"_C","_tau_",taus[2],".rds"))
timings(copula=G.cop.d5,name= paste0("G","_tau_",taus[2]),
       file=paste0("timing","_dim_",d.5,"_G","_tau_",taus[2],".rds"))
timings(copula=NG.d23,name=paste0("NG23_tau_",paste0(taus, collapse = "_")),
       file=paste0("timing","_dim_",sum(ds.5),"_NG23","_tau_",
                   paste0(taus, collapse = "_",".rds")))

## Copulas from Section 1.4 above
timings(copula=t.cop.d10,name = paste0("t",nu,"_tau_",taus[2]),
       file=paste0("timing","_dim_",d.10,"_t",nu,"_tau_",taus[2],".rds"))
timings(copula=C.cop.d10,name= paste0("C","_tau_",taus[2]),
       file=paste0("timing","_dim_",d.10,"_C","_tau_",taus[2],".rds"))
timings(copula=G.cop.d10,name= paste0("G","_tau_",taus[2]),
       file=paste0("timing","_dim_",d.10,"_G","_tau_",taus[2],".rds"))
timings(copula=NG.d55,name=paste0("NG55_tau_",paste0(taus, collapse = "_")),
       file=paste0("timing","_dim_",sum(ds.10),"_NG55","_tau_",
                   paste0(taus, collapse = "_",".rds")))
