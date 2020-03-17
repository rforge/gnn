## By Marius Hofert and Avinash Prasad

## Demo for measuring computation time for certain examples in
## Hofert, Prasad, Zhu ("Quasi-random sampling for
## multivariate distributions via generative neural networks"). The NNs were
## trained on an NVIDIA Tesla P100 GPU.

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # note: access of functions via '::' fails for this package
## => would allow to set the seed with use_session_with_seed(271), but then no GPU or CPU parallelism
tf_version() # dummy call to activate connection to TensorFlow (any first call will fail on the cluster; here: NULL)
use_virtualenv(Sys.getenv('VIRTUAL_ENV')) # tensorflow command to access the activated Python environment
if(packageVersion("qrmtools") < "0.0.11")
  stop('Consider updating via install.packages("qrmtools", repos = "http://R-Forge.R-project.org")')
library(qrmtools) # for ES_np()
if(packageVersion("qrng") < "0.0.7")
  stop('Consider updating via install.packages("qrng", repos = "http://R-Forge.R-project.org")')
library(qrng) # for sobol()
if(packageVersion("copula") < "0.999.19")
  stop('Consider updating via install.packages("copula", repos = "http://R-Forge.R-project.org")')
library(copula) # for the considered copulas
library(gnn) # for the used GMMN models
library(microbenchmark) ## For measuring time

## Global training parameters
package <- NULL # uses pre-trained NNs from 'gnn' (recommended); for retraining, use package = NULL
dim.hid <- 300L # dimension of the (single) hidden layer
ntrn <- 60000L # training dataset size (number of pseudo-random numbers from the copula)
nbat <- 5000L # batch size for training (number of samples per stochastic gradient step)
nepo <- 300L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
stopifnot(dim.hid >= 1, ntrn >= 1, 1 <= nbat, nbat <= ntrn, nepo >= 1)

## Other global parameters
ngen <- 10000L # sample size of the generated data
ncores <- 1 # detectCores() # number of cores to be used for parallel computing
stopifnot(ncores == 1) # as of 2019, TensorFlow does not allow multicore calculations in R
## Note: See the discussion on https://stat.ethz.ch/pipermail/r-sig-hpc/2019-August/002092.html

##' @title Measuring GMMN training time
##' @param copula copula object
##' @param name character string (copula and taus together) for trained GMMNs
##' @return List containing trained GMMN and elapsed training time TODO: Correct to only report elapsed time? 
training_time <- function(copula,name){
  ## Generate training data
  set.seed(271) # for reproducibility
  U <- rCopula(ntrn, copula = copula) # generate training dataset from a PRNG
  ## Train
  dim.in.out <- dim(copula) # = dimension of the prior distribution fed into the GMMN
  NNname <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,
                   "_nbat_",nbat,"_nepo_",nepo,"_",name,".rda")
  GNN <- GMMN_model(c(dim.in.out, dim.hid, dim.in.out)) # model setup
  ## Have to use train() instead of train_once() to capture only training time. 
  ## Additionally if saved NN exists in local repo or if package='gnn', train_once()
  ## will lead to misleading train times. Hence we use separately train and save. 
  train.time <- human_time(GMMN <- train(GNN, data = U,
                                         batch.size = nbat, nepoch = nepo))["elapsed"] 
  ## Saving GMMNs
  save_rda(to_savable(GMMN),file=NNname,names=rm_ext(basename(NNname)))
  list(GMMN=GMMN,train.time=train.time) # Return GMMN along with GMMN training time
}

##' @title Measuring sampling times for PRNG, QRNG, GMMN PRNG and GMMN QRNG sampling 
##' @param copula copula object
##' @param GMMNmod GMMN model trained on pseudo-random samples from 'copula'
##' @return list of (average) sampling times for each of the four sampling procedures 
sampling_times <- function(copula,GMMNmod){
  reps <- 100 # Number of times we repeat the sampling exercise to guage computing time
  d <- dim(copula) # Dimension of copula
  ## Measuring sampling times for PRNG, GMMN PRNG, QRNG and GMMN QRNG 
  ## using microbenchmark. The sampling is repeated for reps to better gauge computing time. Since all
  ## execution times are reported in nanoseconds we divide by 10^9 to convert into seconds. 
  
  ## 1) Copula PRNG
  PRNG.time <- paste0(mean(microbenchmark(rCopula(ngen, copula = copula),times=reps)$time/10^9),"s")
  ## 2) GMMN PRNG
  GMMN.PRNG.time <- paste0(mean(microbenchmark(predict(GMMNmod, 
                                                x = matrix(rnorm(ngen * d), ncol = d)),times=reps)$time/10^9),"s") 
  ## 3) Copula QRNG
  ## If available in analytical form, draw from a real QRNG
  cCopula.inverse.avail <- is(copula, "normalCopula") || is(copula, "tCopula") ||
    is(copula, "claytonCopula")
  if(cCopula.inverse.avail){
      QRNG.time <- paste0(mean(microbenchmark(cCopula(sobol(ngen, d = d, randomize = "Owen", seed = 271), 
                          copula = copula, inverse = TRUE),times=reps)$time/10^9),"s")
  } else { # For cases where we cannot generate QRNG samples via CDM method
    QRNG.time <- "NA"
  }
  ## 4) GMMN QRNG
    GMMN.QRNG.time <- paste0(mean(microbenchmark(predict(GMMNmod, 
                          x = qnorm(sobol(ngen, d = d, randomize = "Owen", seed = 271))),times=reps)$time/10^9),"s")

  list(PRNG.time=PRNG.time,GMMN.PRNG.time=GMMN.PRNG.time,QRNG.time=QRNG.time,GMMN.QRNG.time=GMMN.QRNG.time)
  }

##' @title Measuring computing time for GMMN training
##'  and computing times for  PRNG, QRNG, GMMN PRNG and GMMN QRNG sampling 
##' @param copula copula object
##' @param name character string (copula and taus together) for trained GMMNs
##' @param file A character string specifying the file name used to save results
##' @return list containing GMMN training time and sub-list 
##' containing (average) computing times for each of the four sampling procedures 
computing_times <- function(copula,name,file){
  if(file.exists(file)){
    ctimes <- readRDS(file)  
  }
  else {
  ## Perform GMMN training and obtain trained GMMN along with training time  
  training <- training_time(copula=copula,name=name)
  GMMNmod <- training$GMMN[["model"]]
  ## Measuring sampling times for PRNG, GMMN PRNG, GMMN QRNG, QRNG procedures
  sampling.times <- sampling_times(copula=copula,GMMNmod=GMMNmod)
  comp.times <- list(train.time=training$train.time,sampling.times=sampling.times)
  saveRDS(comp.times,file=file)
  }
  comp.times
}

### 1 Copulas we use ###########################################################

tau <- 0.5

### 1.1 d = 2 ##################################################################

d <- 2 # copula dimension

## t copula
nu <- 4 # degrees of freedom of the t copulas
th.t <- iTau(tCopula(), tau = tau) # parameter
t.cop.d2 <- tCopula(th.t, dim = d, df = nu)

## Clayton copula
th.C <- iTau(claytonCopula(), tau = tau)
C.cop.d2 <- claytonCopula(th.C, dim = d)

### 1.2 d = 5 ##################################################################

d. <- 5 # copula dimension

## t copula
th.t <- iTau(tCopula(), tau = tau) # parameter
t.cop.d5 <- tCopula(th.t, dim = d., df = nu)

## Clayton copula
th.C <- iTau(claytonCopula(), tau = tau)
C.cop.d5 <- claytonCopula(th.C, dim = d.)

### 1.3 d = 10 ##################################################################

d.. <- 10 # copula dimension

## t copula
th.t <- iTau(tCopula(), tau = tau) # parameter
t.cop.d10 <- tCopula(th.t, dim = d.., df = nu)

## Clayton copula
th.C <- iTau(claytonCopula(), tau = tau)
C.cop.d10 <- claytonCopula(th.C, dim = d..)

### 2 Train the GMMNs from a PRNG of the respective copula and measure computing times 
### for training and sampling.

## Copulas from Section 1.1 above
computing_times(copula=t.cop.d2,name = paste0("t",nu,"_tau_",tau),
                file=paste0("computing_times","_dim_",d,"_t",nu,"_tau_",tau,".rds"))
computing_times(copula=C.cop.d2,name= paste0("C","_tau_",tau),
                file=paste0("computing_times","_dim_",d,"_C","_tau_",tau,".rds"))

## Copulas from Section 1.2 above
computing_times(copula=t.cop.d5,name = paste0("t",nu,"_tau_",tau),
                file=paste0("computing_times","_dim_",d.,"_t",nu,"_tau_",tau,".rds"))
computing_times(copula=C.cop.d5,name= paste0("C","_tau_",tau),
                file=paste0("computing_times","_dim_",d.,"_C","_tau_",tau,".rds"))

## Copulas from Section 1.3 above
computing_times(copula=t.cop.d10,name = paste0("t",nu,"_tau_",tau),
                file=paste0("computing_times","_dim_",d..,"_t",nu,"_tau_",tau,".rds"))
computing_times(copula=C.cop.d10,name= paste0("C","_tau_",tau),
                file=paste0("computing_times","_dim_",d..,"_C","_tau_",tau,".rds"))