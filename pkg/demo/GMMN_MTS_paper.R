
## Code to reproduce the results of Hofert, Prasad, Zhu
#("Multivariate time series modeling with generative neural networks").


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # interface to TensorFlow (numerical computation with tensors)
library(qrmtools) # for ES_t(), ARMA_GARCH_fit
if(packageVersion("copula") < "0.999.19")
  stop('Consider updating via install.packages("copula", repos = "http://R-Forge.R-project.org")')
library(copula) # for the considered copulas
library(gnn) # for the used GMMN models
library(xts) # for na.fill
library(MASS)
library(qrmdata) ## For required datasets
library(rugarch) ## For GARCH fit 
library(scoringRules) ## For vs_sample()
doPDF<-require(crop)
###   Defining colors#######################
library(RColorBrewer)
pal <- colorRampPalette(c("#000000", brewer.pal(8, name =
                                                  "Dark2")[c(7, 3, 5, 4, 6)])) # function
cols <- pal(8) # get colors from that palette
## Global training parameters 
package<-NULL
nepo<-1000L
### 0 Auxiliary functions ######################################################

### 0.1 Retrieving financial time series data  ##################################################

##' @title Retrieve financial time series data using qrmtools package with appropriate transformation applied
##' @param type.series Character string describing which financial time series to retrieve. Choices 
##' include "US_exchange_rates", "GBP_exchange_rates", "US_interest_rates" and "CA_interest_rates" #TODO: Maybe more
##'  @param train.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for training period
##'  @param test.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for test period
##'  @param ...  Can choose type of transform applied to raw financial time series to obtain series of risk-factor changes,
##'  e.g log-returns (default) by providing method for returns function. 
##'  @return list of length 2 containing return series during training and test period. 
get_ts <- function(type.series,train.period,test.period,...){
  if (grepl("exchange_rates",type.series)&grepl("US",type.series)){
    data("CAD_USD","GBP_USD","EUR_USD","CHF_USD","JPY_USD")
    raw <- cbind(CAD_USD,GBP_USD,EUR_USD,CHF_USD,JPY_USD)
  }
  else if (grepl("exchange_rates",type.series) & grepl("GBP",type.series)){
    data("CAD_GBP","USD_GBP","EUR_GBP","CHF_GBP","JPY_GBP","CNY_GBP")
    raw <- cbind(CAD_GBP,USD_GBP,EUR_GBP,CHF_GBP,JPY_GBP,CNY_GBP)
  }
  else if (grepl("interest_rates",type.series) & grepl("US",type.series)){
    data ("ZCB_USD")
    raw <- ZCB_USD
  }
  else if (grepl("interest_rates",type.series) &grepl("CA",type.series)){
    data ("ZCB_CAD")
    raw <- ZCB_CAD/100
  }
  else {stop ("Wrong 'type.series'")
  }
  ## Extract the original (price) time series for both training and test period
  pseries.train <- as.matrix(raw[paste0(train.period,collapse="/")])
  pseries.test <- as.matrix(raw[paste0(test.period,collapse="/")])
  ## Convert to risk-factor changes (return) series for both training and test periods
  rseries.train <- returns(pseries.train,...)
  # Note the first return in the test period is based on change in price from last day of training period
  rseries.test <- returns(rbind(tail(pseries.train,n=1),pseries.test),...)
  
  res<-list(train=rseries.train,test=rseries.test)
}

### 0.2 Modeling marginal time series using ARMA-GARCH ##################################################

##' @title Marginal time series modeling using ARMA-GARCH
##' @param data matrix containing time series data (i.e. training data)
##' @param file character string ending in .rds containing the file name used to save the fitted models 
##' @param garch.order integer vector of length 2 containing the GARCH orders
##' @param arma.order integer vector of length 2 containing the ARMA orders
##' @param innov.model character string containing the choice of innovaton distribution (by default t) 
##' @param with.mu Logical indicating if we include the 'mu' mean parameter in the ARMA model
##' @return list containing fitted ARMA-GARCH models 
marginal_ts_fit <- function(data, file, garch.order=c(1,1),arma.order=c(1,1),innov.model="std",with.mu=TRUE){
  if (file.exists(file)) {
    fitted.models <- readRDS(file)} 
  else {
    ## Specify the marginal ARMA-GARCH models with the type of GARCH (standard GARCH fixed)
    spec <- rep(list(ugarchspec(variance.model = list(model = "sGARCH", garchOrder = garch.order),
                                mean.model = list(armaOrder = arma.order, include.mean = with.mu),
                                distribution.model = innov.model)),ncol(data))
    ## Returns list of fitted ARMA-GARCH models
    fitted.models <- fit_ARMA_GARCH(x=data, ugarchspec.list = spec,solver='hybrid')
    saveRDS(fitted.models,file=file)
  }
  fitted.models  
}

### 0.3 Modeling cross-sectional dependence  ##################################################

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
    dep.types <- c("norm_ex","t_ex","t_un","gumbel","GMMN")  ## types of dependence models
    dm <- dim(U) 
    n <- dm[1] 
    d <- dm[2]
    ind <- which(sapply(dep.types, function(x) grepl(x, file))) # index of dep.types list that matches somewhere in 'file'
    
    stopifnot(length(ind) == 1) # check if there is only one match
    
    fitted.model <- switch(dep.types[ind],
           "norm_ex" = {
             fitCopula(normalCopula(dim=d), data = U, method = "mpl")
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
           "GMMN"={
             train_once(GMMN_model(GMMN.dim),data=U,batch.size=n,nepoch=nepo,file=file,package=package)
           },
           stop("Wrong 'method'"))
    ## Since train_once already saves GMMN models we only need to save the copula models  
    if (!grepl("GMMN",file)) save_rda(fitted.model,file=file)
  }
  fitted.model  
}

### 0.4 Modeling multivariate time series  ##################################################

##' @title Modeling (3 component) multivariate time series models- wrapper function which fits all dependence models for
##'  a specified dataset
##' @param type.series Character string describing which financial time series to retrieve. Choices 
##' include "US_exchange_rates", "GBP_exchange_rates", "US_interest_rates" and "CA_interest_rates" #TODO: Maybe more
##' @param train.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for training period
##' @param test.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for test period
##' @param with.mu logical indicating if 'mu' parameter is included as part of ARMA model
##' @param pca.dim numeric value specifying number of PCs to use for dimension reduction
##' @return list containing fitted (list of) marginal models, PCA model (if required)
##'  and list of all fitted dependence models (4 parametric copulas and 3 GMMNs) 
all_multivariate_ts_fit <- function(type.series,train.period,test.period,pca.dim=NULL,with.mu=TRUE){
  ## When dealing with interest rate data we have a handful of different specifications
  if (grepl("interest_rates",type.series)){
  ## Grab return series (but not log-returns)
  X <- get_ts(type.series=type.series,train.period=train.period,test.period=test.period,method="diff")  
  ## ARMA-GARCH specification
  with.mu <- FALSE
  } else {
  ## Grab return series
  X <- get_ts(type.series=type.series,train.period=train.period,test.period=test.period)  
  }
  ## Create file name for loading-saving ARMA-GARCH associated with type.series 
  marginal.file <- paste0("ARMA_GARCH_",paste0(train.period,collapse="_"),"_",type.series,".rds")
  ## First we feed training dataset to marginal_ts_fit 
  marginal.models<-marginal_ts_fit(data=X$train,with.mu=with.mu,file=marginal.file) #Fit marginal time series models 
  standard.resid <- lapply(marginal.models$fit, residuals, standardize = TRUE) # grab out standardized residuals from 
  ## Converting residuals to matrix data 
  Y <- as.matrix(do.call(merge, standard.resid))
  ## If dimension reduction is required, apply PCA
  if(!is.null(pca.dim)) {
    PCA.model <- PCA_trafo(Y)
    Y <- PCA.model$PCs[,1:pca.dim]
  } else {
    PCA.model <- NULL
  }
  ## Obtain pseudo-observations
  U <- pobs(Y)
  ## Remove any unnecessary dimension names
  dimnames(U)<-NULL
  ### Create file names for loading-saving various dependence models associated with type.series
  ### and fitting the various dependence models
  dm <- dim(U)
  ntrn <- dm[1]  ## Number of observations in the training dataset
  dim.in.out <- dm[2] ## Dimension of Pseudo-observations used to train dependence models
  ## File names for copula models considered
  file.gumbel <- paste0("fitted_","gumbel","_dim_",dim.in.out,if(!is.null(pca.dim)) "_tpca","_",type.series,".rda")
  file.norm.ex <- paste0("fitted_","norm_ex","_dim_",dim.in.out,if(!is.null(pca.dim)) "_tpca","_",type.series,".rda")
  file.t.ex <- paste0("fitted_","t_ex","_dim_",dim.in.out,if(!is.null(pca.dim)) "_tpca","_",type.series,".rda")
  file.t.un <- paste0("fitted_","t_un","_dim_",dim.in.out,if(!is.null(pca.dim)) "_tpca","_",type.series,".rda")
  ## Fitting the four copula models
  model.gumbel <- dependence_fit(U=U,file=file.gumbel)
  model.norm.ex <- dependence_fit(U=U,file=file.norm.ex)
  model.t.ex <- dependence_fit(U=U,file=file.t.ex)
  model.t.un <- dependence_fit(U=U,file=file.t.un)
  ## GMMN hyperparameters
  dim.hid1 <- 100
  dim.hid2 <- 300
  dim.hid3 <- 600
  ## Set nbat to training data set size
  nbat <- ntrn
  ## File names for GMMN models
  file.G1 <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid1,"_",dim.in.out,"_ntrn_",ntrn,"_nbat_",nbat,
                    "_nepo_",nepo, if(!is.null(pca.dim)) "_tpca","_",type.series,".rda")
  file.G2 <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid2,"_",dim.in.out,"_ntrn_",ntrn,"_nbat_",nbat,
                    "_nepo_",nepo, if(!is.null(pca.dim)) "_tpca","_",type.series,".rda")
  file.G3 <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid3,"_",dim.in.out,"_ntrn_",ntrn,"_nbat_",nbat,
                    "_nepo_",nepo, if(!is.null(pca.dim)) "_tpca","_",type.series,".rda")
  ## Fitting the three GMMN models 
  model.G1 <- dependence_fit(U=U,GMMN.dim=c(dim.in.out,dim.hid1,dim.in.out),file=file.G1)
  model.G2 <- dependence_fit(U=U,GMMN.dim=c(dim.in.out,dim.hid2,dim.in.out),file=file.G2)
  model.G3 <- dependence_fit(U=U,GMMN.dim=c(dim.in.out,dim.hid3,dim.in.out),file=file.G3)
  
  dependence.models=list(model.indep=NULL,model.gumbel=model.gumbel,model.norm.ex=model.norm.ex,
                model.t.ex=model.t.ex,model.t.un=model.t.un,model.G1=model.G1,model.G2=model.G2,model.G3=model.G3)

 list(marginal=marginal.models, PCA=PCA.model,dependence=dependence.models) 
}

### 0.5 Evaluating fitted dependence models using Maximum Mean Discrepancy (MMD) ######################################

##' @title Extract the (underlying) dependence of multivariate time series in the test period as implied by our fitted
##' ARMA-GARCH and PCA models.
##' @param type.series Character string describing which financial time series to retrieve. Choices 
##' include "US_exchange_rates", "GBP_exchange_rates", "US_interest_rates" and "CA_interest_rates" #TODO: Maybe more
##' @param train.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for training period
##' @param test.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for test period
##' @param with.mu logical indicating if 'mu' parameter is included as part of ARMA model
##' @param pca.dim numeric value specifying number of PCs to use for dimension reduction
##' @return A (tau,d*) matrix containing the underlying dependence of the test dataset
extract_dependence_ts <- function(type.series,train.period,test.period,pca.dim=NULL,with.mu=TRUE)
  {
  if (grepl("interest_rates",type.series)){
    ## Grab return series (but not log-returns)
    X <- get_ts(type.series=type.series,train.period=train.period,test.period=test.period,method="diff")  
    ## ARMA-GARCH specification
    with.mu <- FALSE
  } else {
    ## Grab return series
    X <- get_ts(type.series=type.series,train.period=train.period,test.period=test.period)  
  }
  ## Noting the number of observations and dimension of test dataset  
  dm <- dim(X$test)
  n <- dm[1] # number of observations in test data
  d <- dm[2] # dimension of time series data 
  ## Create file name for loading-saving ARMA-GARCH associated with type.series 
  marginal.file <- paste0("ARMA_GARCH_",paste0(train.period,collapse="_"),"_",type.series,".rds")
  ## First we feed testing dataset to marginal_ts_fit 
  marginal.fits<-marginal_ts_fit(data=X$train,with.mu=with.mu,file=marginal.file)$fit #Fit marginal time series models 
  standard.resid <- lapply(marginal.fits, residuals, standardize = TRUE) # grab out standardized residuals from 
  ## Converting residuals to matrix data 
  Y <- as.matrix(do.call(merge, standard.resid))
  dimnames(Y)<-NULL ## Remove any dimension names so as not interfere with  functions like predict()
  ## If dimension reduction is required, apply PCA 
  if(!is.null(pca.dim)) PCA <- prcomp(Y)
  ## Retrieve fitted marginal ARMA-GARCH model specifications 
  specs <- sapply(1:d,function(i) getspec(marginal.fits[[i]]))
  
  for (i in 1:d) setfixed(specs[[i]]) <- coef(marginal.fits[[i]])
  ## Use fitted ARMA-GARCH model to extract model-implied residuals on the test-data
  ## To achieve this we make use of a rolling forecast to obtain the fitted
  ## model implied mean and sigma process into the future (test data).
  test.mean <- sapply(1:d, function (i) 
    fitted(ugarchforecast(specs[[i]],data=X$test[,i],n.ahead=1,n.roll=n-1,out.sample=n-1)))
  
  test.sigma <- sapply(1:d, function (i) 
    sigma(ugarchforecast(specs[[i]],data=X$test[,i],n.ahead=1,n.roll=n-1,out.sample=n-1)))
  Y.test <- (X$test-test.mean)/test.sigma
  ## If we include dimension reduction, we make use of PCA model on training data
  ## to project the residuals of the test data onto the lower dimensional PC space. 
  if (!is.null(pca.dim)) Y.test <- predict(PCA,Y.test)[,1:pca.dim]
  U.test <- pobs(Y.test)
}


##' @title Evaluating the MMD between generated samples from dependence models (copulas, GMMNs) and the underlying 
##' empirical dependence of the test dataset
##' @param generators A list of generators from dependence models (Independence, 4 copulas, 3 GMMNs)
##' @param U.test A matrix containing empirical dependence structure of multivariate time series in test period
##' @param bandwidth numeric (vector) containing the bandwidth parameters (sigma). By default taken to be (0.1,0.3,0.5,0.7,0.9).
##' @return A numeric vector of length 8 which contains mmd values for the corresponding
##'  8 models (independence, 4 copulas, 3 GMMNs)
MMD_eval <- function(generators,U.test,bandwidth=seq(0.1,0.9,by=0.2)){
  ## Specify dimension of test dataset and hence dimension of our generated samples  
  dm <- dim(U.test)
  n <- dm[1]
  d <- dm[2]
  ## We need to convert to tensor objects when we are computing MMD
  U.test <- tf$constant(U.test,dtype=tf$float32)
  
  ## Generate samples from independence, fitted parametric copulas or fitted GMMNs
  U.models <- lapply(1:length(generators),function(i) 
    if (grepl("indep",names(generators)[i])) {
      tf$constant(matrix(runif(n*d),ncol=d),dtype=tf$float32)
    }  else if (grepl("cop",names(generators)[i])){
      tf$constant(rCopula(n,copula=generators[[i]]),dtype=tf$float32)
    }  else if (grepl("G",names(generators)[[i]])){
      tf$constant(pobs(predict(generators[[i]], x = matrix(rnorm(n * d), ncol = d))),dtype=tf$float32) 
    })
  
  ## Compute and return MMD values
   sapply(1:length(U.models),function(i) 
     if(!tf$executing_eagerly()) {
       sess$run(loss(U.models[[i]],U.test,type="MMD",bandwidth=bandwidth))
     } else {
       as.numeric(loss(U.models[[i]],U.test,type="MMD",bandwidth=bandwidth))
     })
}  
  
##' @title Compute B realizations of the MMD metric
##' @param type.series Character string describing which financial time series to retrieve. Choices 
##' include "US_exchange_rates", "GBP_exchange_rates", "US_interest_rates" and "CA_interest_rates" 
##' @param train.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for training period
##' @param test.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for test period
##' @param with.mu logical indicating if 'mu' parameter is included as part of ARMA model
##' @param pca.dim numeric value specifying number of PCs to use for dimension reduction
##' @param B a numeric value specifying the number of realization of the MMD statistic to compute
##' @return A (8,B) matrix containing B realizations for 8 models (independence, 4 copulas, 3 GMMNs)
##' @note When using the function ensure sess<-tf$Session() is specified in the global environment. Once all such
##' computation is finished use   sess$close()
MMD_metric <- function(type.series,train.period,test.period,pca.dim=NULL,with.mu=TRUE,B=100){
    ## Obtain empirical dependence of test data
  U.test <- extract_dependence_ts(type.series=type.series,train.period=train.period,
                                           test.period=test.period,with.mu=with.mu,pca.dim=pca.dim)
  dm <- dim(U.test)
  d <- dm[2] ## Dimension of the empirical dependence structure of test data 
  file <- paste0("MMD","_dim_",d,"_B_",B,"_",type.series,".rds")
  
  if(file.exists(file)){
   mmd.vals <- readRDS(file)  
  }
  else{
  models <- all_multivariate_ts_fit(type.series=type.series,train.period=train.period,
                                    test.period=test.period,with.mu=with.mu,pca.dim=pca.dim)
  ## List of  dependence model generators 
  generators <- list(gen.indep=models$dependence$model.indep,gen.copgumbel=models$dependence$model.gumbel@copula,
                     gen.copnorm.ex=models$dependence$model.norm.ex@copula,gen.copt.ex=models$dependence$model.t.ex@copula,
                 gen.copt.un=models$dependence$model.t.un@copula,gen.G1=models$dependence$model.G1[["model"]],
                 gen.G2=models$dependence$model.G2[["model"]],gen.G3=models$dependence$model.G3[["model"]])
  
  set.seed(271)  ## For reproducibility 
  mmd.vals<-replicate(B,MMD_eval(generators=generators,U.test=U.test))
  saveRDS(mmd.vals,file=file)
  }  
  mmd.vals
}

### 0.6 Forecasting one-period ahead empirical predictive distribution ######################################

##' @title Obtaining (rolling) empirical distribution forecasts with training period fixed (i.e models are not re-fitted).
##' @param n.samples numeric value specifying number of samples (or paths) used to construct empirical distribution forecasts
##' @param n.test   numeric value indicating number of time periods in the testing/evaluation period. 
##' @param margin.model a list containing fitted ARMA-GARCH time series models  
##' @param dep.model a  fitted dependence model generator (either of type of copula or GMMN). 
##' In the independence case, this will be NULL. 
##' @param PCA.model a (fitted) PCA model (see PCA_trafo for type of output)
##' @param initial.ts a (n.test,d)- dimensional matrix containing initial time series values used for the 
##' (rolling) simulation-based forecasts.
##' @param h A numeric value signfying the number of periods ahead for the distribution forecast. Default set to h=1.
##' @param dep.model A character string with the three choices indicating type of dependence model specified. Choices
##' include "GMMN", "copula" or "indep". 
##' @param pca.dim A numeric value specifying the number of PCs used for the dimension reduction. 
##' @return a list of length n.test with each element of the list containing a (h*n.samples,d)-matrix providing the h-day ahead
##' empirical distribution forecasts. 
distribution_forecast_ts <- function(n.samples,n.test,margin.model,dep.model,PCA.model,
                                     initial.ts,pca.dim=NULL,h=1,type.dep=c("GMMN","copula","indep")){
  ## Note the type of dependence model
  type <- match.arg(type.dep)
  ## Grab the fitted marignal models and associated fitted d.o.f for further use
  marginal.fits <- margin.model$fit
  nus.fits <- sapply(marginal.fits, function(x) x@fit$coef[["shape"]]) # fitted d.o.f
  
  ## Grab the required components of the PCA model to be used later
  if (!is.null(pca.dim)){
    Y.pc <- PCA.model$PCs[,1:pca.dim]
    mu.pc <- PCA.model$mu
    Gamma.pc <- PCA.model$Gamma
  }
  ## Dimension of multivariate time series data
  d <- ncol(initial.ts) 
  ## The dimension of dependence model dependence on whether we apply PCA or not 
  if (!is.null(pca.dim))  dim.dep <- pca.dim  else dim.dep <- d
  ## Create list of length equal to number of periods/time steps in the test period 
  returns.dist <- vector('list',length=n.test)
  ## Start looping through test period
  for (i in 1:n.test){
    ## Initial d-dimensional returns vector i 
    initial.returns <- initial.ts[i,]
    if (i==1){
      ## For the first period (timestep), we set the initial conditional sigma and residuals 
      ## using out fitted (on the training dataset) marginal models 
      initial.sigma <- sapply(1:d, function (j) tail(marginal.fits[[j]]@fit$sigma,n=1))
      initial.residual <- sapply(1:d, function (j) tail(marginal.fits[[j]]@fit$residuals,n=1))     
    }
    else{
      ## Note that the sigmaSim found in the temp.sim object (from the previous timepoint i-1) gives us a (h,n.sample)-sized matrix 
      ## As we re-compute a new distribution forecast in each time point (day) we just need the first row of
      ## the given matrix. Moreover the first row of this matrix will always be a constant across all n.sample values
      ## Thus we can just choose the first value. This is naturally repeated across all d dimensions. 
      ## This also generalizes for h>1, since even in that case we may want to generate a new h-period ahead distribution forecast
      ## starting from each point i in the test/evaluation period. That is this is suited for a sort of rolling h-day ahead forecast
      ## that is re-computed at the end of each period for the next h-periods. In that case, taking the first value of 
      ## this matrix suffices as the first row will also be constant even when temp.sim object is for h>1. Thus this
      ## d-dimensional conditional sigma  is used as a initialized for simulated the predictive distribution for the next day. 
      initial.sigma <- sapply(1:d, function(j) temp.sim[[j]]@simulation$sigmaSim[1]) 
      ## To obtain the initial residual at timepoint i that is based on the "realized" return values
      ## in the test point, we need to go about this in a round about way. We first calculate the conditional mean
      ## at time point i-1 using series and residual simulation from the temp.sim object . As before this gives (h,n.sample) matrix.
      # The difference between the seriesSim and residSim is a constant across n.sample values in the first row. As a 
      ## result, it suffices to use the first  value. This is then repeated across all d dimensions. 
      initial.mean <- sapply(1:d, function(j) (temp.sim[[j]]@simulation$seriesSim- temp.sim[[j]]@simulation$residSim)[1])
      initial.residual <- initial.returns-initial.mean
    }
    ## Generating samples from various dependence models 
    switch(type,"GMMN"={
      N01.prior.gen <- matrix(rnorm(n.samples*dim.dep*h),ncol=dim.dep)
      U.sim <- pobs(predict(dep.model,x=N01.prior.gen))
    }
    ,"copula"={U.sim <- rCopula(n.samples*h,copula= dep.model)
    }
    ,"indep"={U.sim <- matrix(runif(n.samples*dim.dep*h),ncol=dim.dep)}
    )
    if (!is.null(pca.dim)){
      ## When we applied PCA, we empirically modeled the marginals distributions of the top pca.dim PCs
      ## So we need to first add back the empirical margins  to our samples from the dependence model. 
      Y.pc.sim <- toEmpMargins(U=U.sim,x=Y.pc)
      Z.sim <- PCA_trafo(x=Y.pc.sim,mu=mu.pc,Gamma=Gamma.pc,inverse=TRUE)
    }
    else{
      ## When we do not apply PCA, we simply add back the (scaled) t marginal distribution
      ## as specified as part of our ARMA-GARCH model 
      Z.sim <- sapply(1:d, function(j)
      sqrt((nus.fits[j]-2)/nus.fits[j]) * qt(U.sim[,j], df = nus.fits[j]))
    }
    ## temp. sim here is an ugarchsim object which contains various components 
    ##(some of which we use to initialize the next step of our recursion)
    temp.sim <- lapply(1:d, function(j)
      ugarchsim(marginal.fits[[j]], 
                n.sim = h, 
                m.sim = n.samples, 
                startMethod = 'sample', 
                presigma=initial.sigma[j], # Feed in the initial conditional sigmas here
                prereturns = initial.returns[j], # Feed in the initial returns here
                preresiduals = initial.residual[j], # Feed in the initial residuals here
                custom.dist = list(name = "sample", # our innovations
                                   distfit = matrix(Z.sim[,j], ncol = n.samples))))
    ## Grab out and record the return distributions at each time point 
    returns.dist[[i]] <- sapply(temp.sim, function(x) fitted(x))
  }
   ## Return a list containing n.test many empirical predictive distributions with each based on n.samples. 
  returns.dist
}

##' @title Wrapper function which computes empirical distribution forecasts for all fitted
##' multivariate time series models 
##' @param type.series Character string describing which financial time series to retrieve. Choices 
##' include "US_exchange_rates", "GBP_exchange_rates", "US_interest_rates" and "CA_interest_rates" #TODO: Maybe more
##' @param train.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for training period
##' @param test.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for test period
##' @param with.mu logical indicating if 'mu' parameter is included as part of ARMA model
##' @param pca.dim numeric value specifying number of PCs to use for dimension reduction
##' @param n.samples A numeric value specfying number of samples used to construct the empirical distribution forecasts.
##' By default this is set to 1000. 
##' @return a list of length 8 corresponding to the 8 dependence models considered (3 GMMNs, 4 copulas and independence). 
##' Each element of this list is in turn a list of type output in distribution_forecast_ts() function. 
##' @note Intermediate results not saved due to size
all_distribution_forecast_ts <- function(type.series,train.period,test.period,pca.dim=NULL,with.mu=TRUE,n.samples=1000,h=1){
  ## Obtain the dataset to work with for type.series and specify different ARMA--GARCH settings
  if (grepl("interest_rates",type.series)){
    ## Grab return series (but not log-returns)
    X <- get_ts(type.series=type.series,train.period=train.period,test.period=test.period,method="diff")
    with.mu <- FALSE
  } else {
    ## Grab return series
    X <- get_ts(type.series=type.series,train.period=train.period,test.period=test.period)  
  }
  ## Note the number of observation in the test dataset
  n.test <- nrow(X$test)
  ## Obtain all fitted multivariate time series models 
  models <- all_multivariate_ts_fit(type.series=type.series,train.period=train.period,
                                    test.period=test.period,with.mu=with.mu,pca.dim=pca.dim)
  ## Grab the three component models 
  margin.models <- models$marginal  ## Marginal
  PCA.model <- models$PCA   ## PCA
  dep.models <- models$dependence  ## Dependence
  ## To create matrix on initial return time series for the recursive forecasting procedure 
  ## we need to attach the last return from training dataset (initial value for our first period prediction)
  ## and remove the last return in the test dataset (will be initial value for timepoint outside of the test period).
  initial.ts<-rbind(as.numeric(tail(X$train,n=1)),X$test[-c(n.test),])
  set.seed(271) # set seed for reproducibility 
  distr.pred.models <- lapply(1:length(dep.models), function(i) {
    ## Determine type of dependence 
    if (grepl("indep",names(dep.models)[i])) type.dep="indep" else if(grepl("G",names(dep.models)[i])) type.dep="GMMN" else type.dep="copula" 
    distribution_forecast_ts(n.samples=n.samples,n.test=n.test,margin.model=margin.models,
    if(type.dep=="indep") dep.model=dep.models[[i]] else if(type.dep=="copula") dep.model=dep.models[[i]]@copula else if (type.dep=="GMMN") dep.model=dep.models[[i]][["model"]],
      PCA.model=PCA.model,initial.ts=initial.ts,type.dep=type.dep,h=h,pca.dim=pca.dim)
    })
  
    names(distr.pred.models)=c("distr.pred.indep","distr.pred.gumbel","distr.pred.norm.ex",
                             "distr.pred.t.ex","distr.pred.t.un","distr.pred.G1","distr.pred.G2","distr.pred.G3")
  distr.pred.models
}

### 0.7 Evaluating quality of distribution forecasts ######################################


##' @title Evaluating multivariate times series distribution forecasts using various metrics 
##' @param distribution.forecasts A list of distribution forecasts constructed using fixed
##' ARMA-GARCH and PCA model (if required) and various dependence models. A list of type output
##' provided by all_distribution_forecast_ts() function.
##' @param X A list containing the training and test data for a multivariate time series 
##' @param type.metric The various types of forecast evaluation metrics including the variogram score, MSE and 
##' absolute error of portfolio VaR exceedance, i.e abs(actual exceedance - expected exceedance)
##' @param file A character string specifying the file name used to save forecast evaluation metrics
##' @param alpha The alpha used for calculating the portfolio VaR exceedance. Default set to alpha=0.05
##' @param p A numeric value specifying the order of variogram score
##' @return A list of length 8 representing each of the 8 dependence models considered. For each element of this list
##' this function returns a numeric vector of length of test period representing the forecast evaluation metric for each
##' one-period forecast.
distribution_forecast_eval <- function(distribution.forecasts,X,
                                       type.metric=c("variogram_score","MSE","VaR_exceedance_abserror"),
                                       file,alpha=0.05,p=0.5){
  if (file.exists(file)){
    metrics <- readRDS(file)
  } else {
    type <- match.arg(type.metric)
    X.test <- X$test # Grab out test dataset
    dm <- dim(X.test)
    n.test <- dm[1] # Set number of observations in test set
    d.test <- dm[2] # Dimension of test dataset
    n.samples <- nrow(distribution.forecasts[[1]][[1]]) ## The number of samples used to construct the empirical distribution forecast
    switch(type,"variogram_score"={ # Use vs_sample from scoringRules package (p=0.5 by default)
      metrics <- lapply(1:length(distribution.forecasts), 
                        function(j) sapply(1:n.test,
                                           function(i) vs_sample(y=X.test[i,],
                                                                 dat=t(distribution.forecasts[[j]][[i]]),p=p)))
    }
    ,"MSE"={ # (Multivariate) version of MSE calculated using euclidean distances
      metrics <- lapply(1:length(distribution.forecasts), 
                        function(k) sapply(1:n.test,function(i) mean(sapply(1:n.samples,
                                                                            function(j) 
                                                                              dist(rbind(distribution.forecasts[[k]][[i]][j,],X.test[i,]))))))
    }
    ,"VaR_exceedance_abserror"={ ## Portfolio VaR exceedance evaluation metric 
      sum.test <- rowSums(X.test)  ## Compute Aggregate returns for the test data set
      ## Compute Aggregate returns for each path/ sample and each dependence model.
      sum.forecasts <- lapply(1:length(distribution.forecasts), 
                              function(j) sapply(1:n.test, function(i) rowSums(distribution.forecasts[[j]][[i]])))
      
      ## Approximate the alpha quantile based on distribution forecasts of the aggregate sum. 
      sumquantile.forecasts <- lapply(1:length(distribution.forecasts),
                                      function(j) sapply(1:n.test, function(i) as.numeric(quantile(sum.forecasts[[j]][,i],probs=alpha))))
      
      ## Calculate the number of exceedances in the test data at the various quantile levels implied by
      ## each of our distribution forecasts
      exceedance.forecasts <- lapply(1:length(distribution.forecasts), function(j) sum(sum.test<sumquantile.forecasts[[j]])/n.test)
      
      ## The metric we use is a simple absolute error between the expected (alpha) and actual exceedances
      metrics <- lapply(1:length(distribution.forecasts), function(j) abs(exceedance.forecasts[[j]]-alpha))
  } 
  )
    names(metrics) <- names(distribution.forecasts) # Ensure the model names are preserved for the ouput matrix
    saveRDS(metrics,file=file)
  }
  metrics
}

##' @title Producing scatter plots of MMD metrics vs. forecast metrics
##'  where the forecast metric is one of three choices (see type.metric).
##' @param type.series Character string describing which financial time series to retrieve. Choices 
##' include "US_exchange_rates", "GBP_exchange_rates", "US_interest_rates" and "CA_interest_rates" #TODO: Maybe more
##' @param train.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for training period
##' @param test.period Character vector of length 2 consisting of start and end dates in proper date format, i.e 
##'  "YYYY-MM-DD" for test period
##' @type.metric The various types of forecast evaluation metrics including the variogram score, MSE and 
##' absolute error of portfolio VaR exceedance, i.e abs(actual exceedance - expected exceedance)
##' @alpha The alpha used for calculating the portfolio VaR exceedance. Default set to alpha=0.05.
##' @param with.mu logical indicating if 'mu' parameter is included as part of ARMA model
##' @param pca.dim numeric value specifying number of PCs to use for dimension reduction
##' @param n.samples A numeric value specfying number of samples used to construct the empirical distribution forecasts.
##' By default this is set to 1000. 
##' @param B a numeric value specifying the number of realization of the MMD statistic to compute
##' @param p A numeric value specifying the order of variogram score
##' @return Nothing. Produces plots as a side-effect. 
forecast_evaluation_plot <- function(type.series, train.period, test.period,
                                     type.metric,pca.dim=NULL,alpha=0.05,with.mu=TRUE,n.samples=1000,B=100,p=0.5){
  ## Obtain the dataset to work with for type.series and specify ARMA--GARCH in special cases
  if (grepl("interest_rates",type.series)){
    ## Grab return series (but not log-returns)
    X <- get_ts(type.series=type.series,train.period=train.period,test.period=test.period,method="diff")  
    with.mu <- FALSE
  } else {
    ## Grab return series
    X <- get_ts(type.series=type.series,train.period=train.period,test.period=test.period)  
  }
  ## Note the dimension of the dependence to be used in the file name.
  if (!is.null(pca.dim)) dep.dim <- pca.dim else dep.dim <- ncol(X$test)
  ## Obtain the MMD metrics
  MMD.metrics <- MMD_metric(type.series=type.series,
                           train.period=train.period,test.period=test.period,
                           with.mu=with.mu,pca.dim=pca.dim,B=B)
  ## Specify file name for saving/loading forecast metrics 
  filename.metrics <- paste0(type.metric, if(grepl("VaR_exceedance_abserror",type.metric)) "_alpha_", 
                             if(grepl("VaR_exceedance_abserror",type.metric)) alpha,
                             if(grepl("variogram_score",type.metric)) "_p_",
                             if(grepl("variogram_score",type.metric)) p,
                             "_depdim_",dep.dim,"_",type.series,".rds")
  if (file.exists(filename.metrics)){
    forecast.metrics <- readRDS(filename.metrics)
  } else {
    ## Obtain the one-period ahead distribution forecasts.  
    distribution.forecasts <- all_distribution_forecast_ts(type.series=type.series,train.period=train.period,
                                                         test.period=test.period, with.mu=with.mu,
                                                         pca.dim=pca.dim,n.samples=n.samples)
  ## Evaluate the distribution forecasts based on type.metric (The function saves the file in filename.metrics)
  forecast.metrics <- distribution_forecast_eval(distribution.forecasts=distribution.forecasts,
                                                X=X, type.metric=type.metric,
                                                alpha=alpha,file=filename.metrics,p=p)
  }
  ## Compute the average MMD and forecast metric across the test data set
 average.MMD.metrics <- rowMeans(MMD.metrics)
 average.forecast.metrics <- sapply(1:length(forecast.metrics), function(i) mean(forecast.metrics[[i]]))
 names(average.forecast.metrics) <- names(forecast.metrics) ## Preserve model name 
 ## Create a vector of labels with each label corresponding to a dependence model
 labels.vec <- rep(NA,length(forecast.metrics))
 labels.vec[which(grepl("indep",names(average.forecast.metrics)))] <- c("Independent")
 labels.vec[which(grepl("gumbel",names(average.forecast.metrics)))] <- c("Gumbel copula")
 labels.vec[which(grepl("norm.ex",names(average.forecast.metrics)))] <- c("Normal copula (exchangeable)")
 labels.vec[which(grepl("t.ex",names(average.forecast.metrics)))] <- c("t copula (exchangeable)")
 labels.vec[which(grepl("t.un",names(average.forecast.metrics)))] <- c("t copula (unstructured)")
 labels.vec[which(grepl("G1",names(average.forecast.metrics)))] <- c("GMMN model 1")
 labels.vec[which(grepl("G2",names(average.forecast.metrics)))] <- c("GMMN model 2")
 labels.vec[which(grepl("G3",names(average.forecast.metrics)))] <- c("GMMN model 3")
 cols.vec <- c(rep(cols[4],5),rep(cols[7],3))
 pch.vec <- 1:length(average.forecast.metrics)
## Create labels for y-axis depending on the type of forecast evaluation metric. 
if(grepl("MSE",type.metric)) {
  ylabel <- c("AMSE")
} else if (grepl("variogram_score",type.metric)) {
    ylabel <- substitute("AVS"^p.,list(p.=p))
} else if (grepl("VaR_exceedance_abserror",type.metric)){
    ylabel <- substitute("VEAR"[alpha.],list(alpha.=1-alpha))
}  
## Plotting average MMD vs. average forecast evaluation metric 
filename.plot <- paste0("fig_MMD","_vs_",rm_ext(filename.metrics),".pdf")
if(doPDF)
  pdf(file = filename.plot, bg = "transparent")
plot(average.MMD.metrics,average.forecast.metrics,xlab='AMMD',ylab=ylabel,pch=pch.vec,col=cols.vec)
legend(x="topleft",cex=0.6,legend= labels.vec, bty='n',pch=pch.vec,col=cols.vec)
if(doPDF) dev.off.crop(file)
}

### 1 Producing main results ######################################

if(!tf$executing_eagerly()) sess<-tf$Session() ## Called globally for the MMD calculations 

### 1.1 Producing plots for US exchange rates data with certain training and testing period ######################################

## 1.1.1 Results where MMD is computed using fitted ARMA-GARCH models and PCA models projected onto test dataset

## Obtain results for all MSE, variogram score and VaR exceedance absolute error (with alpha=0.05) evaluation metrics 
forecast_evaluation_plot(type.series="US_exchange_rates", train.period=c("2000-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="MSE")
forecast_evaluation_plot(type.series="US_exchange_rates", train.period=c("2000-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="variogram_score",p=0.25)
forecast_evaluation_plot(type.series="US_exchange_rates", train.period=c("2000-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="VaR_exceedance_abserror")

### 1.2 Producing plots for GBP exchange rates data with certain training and testing period ######################################

## 1.2.1 Results where MMD is computed using fitted ARMA-GARCH models and PCA models projected onto test dataset

## Obtain results for all MSE, variogram score and VaR exceedance absolute error (with alpha=0.05) evaluation metrics 

forecast_evaluation_plot(type.series="GBP_exchange_rates", train.period=c("2000-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="MSE")
forecast_evaluation_plot(type.series="GBP_exchange_rates", train.period=c("2000-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="variogram_score",p=0.25)
forecast_evaluation_plot(type.series="GBP_exchange_rates", train.period=c("2000-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="VaR_exceedance_abserror")


### 1.3 Producing plots for US interest rates data with certain training and testing period ######################################

## For this data we fix the PCA dimension to be 3. Original dimension is 30. 

## 1.3.1 Results where MMD is computed using fitted ARMA-GARCH models and PCA models projected onto test dataset
## Obtain results for only MSE, variogram score evaluation metrics 

## pca.dim=3
forecast_evaluation_plot(type.series="US_interest_rates", train.period=c("1995-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="MSE",pca.dim=3)
forecast_evaluation_plot(type.series="US_interest_rates", train.period=c("1995-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="variogram_score",pca.dim=3,p=0.25)


### 1.4 Producing plots for US interest rates data with certain training and testing period ######################################
## For this data we experiment with PCA dimension equal to 4. Original dimension of data is 120.  

## 1.4.1 Results where MMD is computed using fitted ARMA-GARCH models and PCA models projected onto test dataset

## pca.dim=4
forecast_evaluation_plot(type.series="CA_interest_rates", train.period=c("1995-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="MSE",pca.dim=4)
forecast_evaluation_plot(type.series="CA_interest_rates", train.period=c("1995-01-01", "2014-12-31"),
                         test.period=c("2015-01-01", "2015-12-31"),type.metric="variogram_score",pca.dim=4,p=0.25)

if(!tf$executing_eagerly()) sess$close()


