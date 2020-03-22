## By Marius Hofert and Avinash Prasad

## Code to reproduce data example presented in
## Hofert, Prasad, Zhu ("Quasi-random sampling for
## multivariate distributions via generative neural networks"). The NNs were
## trained on an NVIDIA Tesla P100 GPU.


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # interface to TensorFlow (numerical computation with tensors)
## => would allow to set the seed with use_session_with_seed(271), but then no GPU or CPU parallelism
if(grepl("gra", Sys.info()[["nodename"]])) {
  tf_version() # dummy call to activate connection to TensorFlow (any first call will fail on the cluster; here: NULL)
  use_virtualenv(Sys.getenv('VIRTUAL_ENV')) # tensorflow command to access the activated Python environment
}
library(qrmtools) # for ES_t(), ARMA_GARCH_fit
if(packageVersion("copula") < "0.999.19")
  stop('Consider updating via install.packages("copula", repos = "http://R-Forge.R-project.org")')
library(copula) # for the considered copulas
library(gnn) # for the used GMMN models
library(xts) # for na.fill
library(MASS)
library(qrmdata) ## For required datasets
library(rugarch) ## For GARCH fit 
###   Defining colors#######################

## Global training parameters 
package<-NULL
nepo<-300L
ngen <- 10000L # sample size of the generated data
## GMMN hyperparameter
dim.hid <- 300

### 0 Auxiliary functions ######################################################


### 0.1 Modeling marginal time series using ARMA-GARCH ##################################################

##' @title Marginal time series modeling using ARMA-GARCH
##' @param data matrix containing time series data (i.e. training data)
##' @param file character string ending in .rds containing the file name used to save the fitted models 
##' @param garch.order integer vector of length 2 containing the GARCH orders
##' @param arma.order integer vector of length 2 containing the ARMA orders
##' @param innov.model character string containing the choice of innovaton distribution (by default t) 
##' @return list containing fitted ARMA-GARCH models 
marginal_ts_fit <- function(data, file, garch.order=c(1,1),arma.order=c(1,1),innov.model="std"){
  if (file.exists(file)) {
    fitted.models <- readRDS(file)
    } else {
    ## Specify the marginal ARMA-GARCH models with the type of GARCH (standard GARCH fixed)
    spec <- rep(list(ugarchspec(variance.model = list(model = "sGARCH", garchOrder = garch.order),
                                mean.model = list(armaOrder = arma.order),
                                distribution.model = innov.model)),ncol(data))
    ## Returns list of fitted ARMA-GARCH models
    fitted.models <- fit_ARMA_GARCH(data, ugarchspec.list = spec,solver='hybrid')
    saveRDS(fitted.models,file=file)
  }
  fitted.models  
}

### 0.2 Modeling cross-sectional dependence  ##################################################

##' @title Modeling dependence using copulas/GMMNs
##' @param U matrix of pseudo-observations (training data)
##' @param GMMN.dim numeric vector of length at least two giving the dimensions of
##' the input layer, the hidden layer(s) (if any) and the output layer. Only needed if fitting GMMNs. 
##' @param file character string (with ending .rda) specifying the file
##' to save the results in. 
##' @return  fitted copula or GMMN models 
dependence_fit <- function(U,GMMN.dim,file){
  if (exists_rda(file, names = rm_ext(basename(file)), package = package)){
    fitted.model <- read_rda(file=file,names=rm_ext(basename(file)))
    ## Only for GMMN models we need to additionally use to_callable to unserialize models 
    if (grepl("GMMN",file)) fitted.model <- to_callable(fitted.model)
  } else{ ## Fitting and saving copula (four types in particular) or GMMN models 
    dep.types <- c("norm_ex","norm_un","t_ex","t_un","clayton","gumbel","GMMN")  ## types of dependence models
    dm <- dim(U) 
    n <- dm[1] 
    d <- dm[2]
    ind <- which(sapply(dep.types, function(x) grepl(x, file))) # index of dep.types list that matches somewhere in 'file'
    
    stopifnot(length(ind) == 1) # check if there is only one match
    
    fitted.model <- switch(dep.types[ind],
                           "norm_ex" = {
                             fitCopula(normalCopula(dim=d), data = U, method = "mpl")
                           },
                           "norm_un" = {
                             fitCopula(normalCopula(dim=d,dispstr="un"), data = U, method = "mpl")
                           },
                           "t_ex" = {
                             fitCopula(tCopula(dim=d), data = U, method = "mpl")
                           },
                           "t_un"={
                             fitCopula(tCopula(dim=d,dispstr='un'), data = U, method = "mpl")
                           }, 
                           "gumbel"={
                             fitCopula(gumbelCopula(dim=d), data = U, method = "mpl") 
                           },
                           "clayton"={
                             fitCopula(claytonCopula(dim=d), data = U, method = "mpl") 
                           },
                           "GMMN"={
                             train_once(GMMN_model(GMMN.dim),data=U,batch.size=n,nepoch=nepo,file=file,package=package)
                           },
                           stop("Wrong 'method'"))
    ## Since train_once already saves GMMN models we only need to save the copula models  
    if (!grepl("GMMN",file)) save_rda(fitted.model,file=file)
  }
  fitted.model  
}

### 0.3 Modeling multivariate time series  ##################################################

##' @title Modeling multivariate time series models- wrapper function which fits all dependence models for
##'  a specified dataset
##' @param X matrix containing time series data (i.e. training data)
##' @param series.info list of 2 character strings specifying training period and type of dataset
##' @return list containing fitted (list of) marginal models, matrix of pseudo-observations which represents training
##' data for dependence models and list of all fitted dependence models (6 parametric copulas and one GMMN model) 
multivariate_ts_fits <- function(X,series.info){
  ## Create file name for loading-saving ARMA-GARCH models
  marginal.file <- paste0("ARMA_GARCH_",paste0(series.info[1],collapse="_"),"_",series.info[2],".rds")
  ## First we feed training dataset to marginal_ts_fit 
  marginal.models<-marginal_ts_fit(X,file=marginal.file) #Fit marginal time series models 
  standard.resid <- lapply(marginal.models$fit, residuals, standardize = TRUE) # grab out standardized residuals from 
  ## Converting residuals to matrix data 
  Y <- as.matrix(do.call(merge, standard.resid))
  ## Obtain pseudo-observations
  U <- pobs(Y)
  ## Remove any unnecessary dimension names
  dimnames(U)<-NULL
  ### Create file names for loading-saving various dependence models
  ### and fitting the various dependence models
  dm <- dim(U)
  ntrn <- dm[1]  ## Number of observations in the training dataset
  dim.in.out <- dm[2] ## Dimension of Pseudo-observations used to train dependence models

  ## File names for copula models considered
  file.gumbel <- paste0("copula_","gumbel","_dim_",dim.in.out,"_",series.info[2],".rda")
  file.clayton <- paste0("copula_","clayton","_dim_",dim.in.out,"_",series.info[2],".rda")
  file.norm.ex <- paste0("copula_","norm_ex","_dim_",dim.in.out,"_",series.info[2],".rda")
  file.norm.un <- paste0("copula_","norm_un","_dim_",dim.in.out,"_",series.info[2],".rda")
  file.t.ex <- paste0("copula_","t_ex","_dim_",dim.in.out,"_",series.info[2],".rda")
  file.t.un <- paste0("copula_","t_un","_dim_",dim.in.out,"_",series.info[2],".rda")
  ## Fitting the four copula models
  model.gumbel.cop <- dependence_fit(U,file=file.gumbel)
  model.clayton.cop <- dependence_fit(U,file=file.clayton)
  model.norm.ex.cop <- dependence_fit(U,file=file.norm.ex)
  model.norm.un.cop <- dependence_fit(U,file=file.norm.un)
  model.t.ex.cop <- dependence_fit(U,file=file.t.ex)
  model.t.un.cop <- dependence_fit(U,file=file.t.un)

  ## Set nbat to training data set size
  nbat <- ntrn
  ## File names for GMMN model
  file.GMMN <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",dim.in.out,"_ntrn_",ntrn,"_nbat_",nbat,
                    "_nepo_",nepo,"_",series.info[2],".rda")
  ## Fitting GMMN model 
  model.GMMN <- dependence_fit(U,GMMN.dim=c(dim.in.out,dim.hid,dim.in.out),file=file.GMMN)
  
  dependence.models=list(model.gumbel.cop=model.gumbel.cop,model.clayton.cop=model.clayton.cop,
                         model.norm.ex.cop=model.norm.ex.cop,model.norm.un.cop=model.norm.un.cop,
                         model.t.ex.cop=model.t.ex.cop,model.t.un.cop=model.t.un.cop,model.GMMN=model.GMMN)
  
  list(marginal=marginal.models,U.train=U,dependence=dependence.models) 
}


### 0.4 Evaluating fitted dependence models using two-sample gof stats ######################################

##' @title Compute B realizations of two-sample gof stats
##' @param U.train matrix containing pseudo-observations (i.e. training data)
##' @param dep.models list of all fitted dependence models (6 parametric copulas and one GMMN model)
##' @param series.info list of 2 character strings specifying training period and type of dataset
##' @param B a numeric value specifying the number of realization of the two sample gof statistic to compute
##' @return (B, 7)-matrix containing the 100 replications of the Cramer-von Mises
##'         statistic for the 7 competing dependence models
gof_2stats <- function(U.train,dep.models,series.info,B=100){
  dm <- dim(U.train)  
  n <- dm[1]  # Number of observations in training data
  d <- dm[2]  # Dimension of training data
  
  ## File name for loading-saving realizations of gof 2 sample test statistics
  file <- paste0("gof2stat","_dim_",d,"_ngen_",ngen,"_B_",B,"_",series.info[2],".rds")
  if(file.exists(file)){
    gof.stats <- readRDS(file)  
  }else {

    ## Auxiliary function
    aux <- function() { 
      ## Generate samples from fitted parametric copulas or fitted GMMN
      U.models <- lapply(1:length(dep.models),function(i) 
         if (grepl("cop",names(dep.models)[i])){
          pobs(rCopula(ngen,copula=dep.models[[i]]@copula))
        }  else if (grepl("GMMN",names(dep.models)[i])){
          pobs(predict(dep.models[[i]][["model"]], x = matrix(rnorm(ngen * d), ncol = d)))
        })
      ## Compute the gof 2 sample test statistics for each of the generated samples
     sapply(1:length(U.models), function(i) gofT2stat(U.models[[i]],U.train))
    }
    ## Compute B realizations of gof 2 sample test statistics
    set.seed(271)  ## For reproducibility 
    gof.stats <- t(replicate(B,aux()))
    colnames(gof.stats) <- names(dep.models) # Preserve model names
    saveRDS(gof.stats,file=file)
  }  
  gof.stats
}

##' @title Wrapper function for Producing scatter plots (for visualization) and
##'  boxplots (for assessing fit of dependence models)
##' @param X matrix containing time series data (i.e. training data)
##' @param series.info list of 2 character strings specifying training period and type of dataset
##' @param B a numeric value specifying the number of realization of the two sample gof statistic to compute
##' @return Nothing. Produces plots by side-effect. 
plots_wrapper1 <- function(X, series.info,B=100){
  ## First fit multivariate time series models and retreive pseudo-observations and 
  ## and list of all fitted dependence models
  raw <- multivariate_ts_fits(X,series.info=series.info)
  U.train <- raw$U.train 
  dep.models <- raw$dependence
  
  dm <- dim(X)  
  n <- dm[1]  # Number of observations in training data
  d <- dm[2]  # Dimension of training data
  ## Scatter plot to visualize pseudo-observations (after removing serial dependence)
  file.scatter <- paste0("fig_scatter_dim_",d,"_",series.info[2],".pdf")
  doPDF <- require(crop)
  if(doPDF) pdf(file = file.scatter)
  pairs2(U.train,pch=".")
  if(doPDF) dev.off.crop(file.scatter)
  
  ## Compute two-sample gof test statistic 
  gof.stats <- gof_2stats(U.train,dep.models=dep.models,series.info=series.info,B=B)  
  ## Produce a boxplot of replications of the two sample gof statistic
  ## File name for boxplot
  file.bp <-  paste0("fig_gof2stat","_dim_",d,"_ngen_",ngen,"_B_",B,"_",series.info[2],".pdf")
  
  ## Create a vector of names with each names corresponding to a fitted dependence model
   names.vec <- rep(NA,ncol(gof.stats))
  names.vec[which(grepl("gumbel",colnames(gof.stats)))] <- "Gumbel"
  names.vec[which(grepl("clayton",colnames(gof.stats)))] <- "Clayton"
  names.vec[which(grepl("norm.ex",colnames(gof.stats)))] <- "Normal (ex)"
  names.vec[which(grepl("norm.un",colnames(gof.stats)))] <- "Normal (un)"
  names.vec[which(grepl("t.ex",colnames(gof.stats)))] <- "t (ex)"
  names.vec[which(grepl("t.un",colnames(gof.stats)))] <- "t (un)"
  names.vec[which(grepl("GMMN",colnames(gof.stats)))] <- "GMMN"
  doPDF <- require(crop)
  if(doPDF) pdf(file = file.bp,height = 9.5,width=9.5)
  boxplot(gof.stats,names=names.vec,ylab=expression(S[list(n[gen],n[trn])]),log="y")
  mtext(substitute(B.~"replications, d ="~d.~", "~n[gen]~"="~ngen.~", "~n[trn]~"="~ntrn.,
                  list(B. = B, d.=d,ngen.=ngen,ntrn.=n)),
      side = 4, line = 0.5, adj = 0)
  if(doPDF) dev.off.crop(file.bp)
}



### 1 Retrieve financial time series data using qrmtools package ######################################

## Loading S&P 500 dataset 
data("SP500_const") # load the SP 500 constituents data from qrmdata
train.period <- c("1995-01-01", "2015-12-31") # training time period
S.SP <- SP500_const[paste0(train.period, collapse = "/"),] # data
## Filter data for constiutents with very little NAs in their history and fill the NAs
keep <- apply(S.SP, 2, function(x.) mean(is.na(x.)) <= 0.01) # keep constituents with <= 1% NA
S.SP <- S.SP[, keep] # data we keep
S.SP <- na.fill(S.SP, fill = "extend") # fill NAs
X.SP <- returns(S.SP) # compute negative log-returns


## 1.1 3 Tech and 2 Financial sector stocks #####################
tickers.T3F2 <- c("AAPL","MSFT","IBM","BAC","C")
X.SP5.T3F2 <- X.SP[,tickers.T3F2]  

## 1.2 3 Tech, 3 Financial and 4 Industrial sector stocks #####################
tickers.T3F3I4 <- c("INTC","ORCL","IBM","COF","JPM","AIG","MMM","BA","GE","CAT")
X.SP10.T3F3I4 <- X.SP[,tickers.T3F3I4]

### 2 Produce plots to visualize training data and assess goodness of dependence fit ##############################################

## For SP500 constituents specified in 1.1
plots_wrapper1(X.SP5.T3F2,series.info=c(paste0(train.period,collapse="_"),
                                      paste0("SP500_",paste0(tickers.T3F2,collapse = "_"))))

## For SP500 constituents specified in 1.2
plots_wrapper1(X.SP10.T3F3I4,series.info=c(paste0(train.period,collapse="_"),
                                        paste0("SP500_",paste0(tickers.T3F3I4,collapse = "_"))))



