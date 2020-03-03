## By Marius Hofert and Avinash Prasad

## Reproducing R script for Hofert, Prasad, Zhu ("Multivariate time
## series modeling with generative neural networks"). The code was run on an
## NVIDIA Tesla P100 GPU in about 7h.


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # interface to TensorFlow (numerical computation with tensors)
library(qrmtools) # for ES_t(), ARMA_GARCH_fit
if(packageVersion("copula") < "0.999.19")
    stop('You must update "copula" via install.packages("copula", repos = "http://R-Forge.R-project.org")')
library(copula) # for the considered copulas
library(gnn) # for the used GMMN models
library(xts) # for na.fill
library(MASS)
library(qrmdata) # for required datasets
if(packageVersion("qrmdata") < "2019.12.3.1")
    stop('You must update "qrmdata" via install.packages("qrmdata")')
library(rugarch) # for GARCH fit
library(scoringRules) # for vs_sample()

## Colors
library(RColorBrewer)
pal <- colorRampPalette(c("#000000",
                          brewer.pal(8, name = "Dark2")[c(7, 3, 5, 4, 6)])) # function
cols <- pal(8) # get colors from that palette

## Global training parameters
package <- NULL
nepo <- 1000L


### 0 Auxiliary functions ######################################################

### 0.1 Retrieving financial time series data  #################################

##' @title Retrieve Financial Time Series Data
##' @param type.series character string specifying the financial time series to
##'        retrieve. Choices include FX_USD, FX_GBP, ZCB_USD, ZCB_CAD.
##' @param train.period character vector of length 2 with entries "YYYY-MM-DD"
##'        specifying the start and end date of the training period
##' @param test.period same for the test period
##' @param ... type of transform applied to raw financial time series to obtain
##'        series of risk-factor changes; see argument 'method' of returns()
##' @return 2-list containing return series during training and test period
get_ts <- function(type.series, train.period, test.period, ...)
{
    raw <- if (grepl("FX", type.series) & grepl("USD", type.series)) {
               data("CAD_USD", "GBP_USD", "EUR_USD", "CHF_USD", "JPY_USD")
               cbind(CAD_USD, GBP_USD, EUR_USD, CHF_USD, JPY_USD)
           } else if (grepl("FX", type.series) & grepl("GBP", type.series)) {
               data("CAD_GBP", "USD_GBP", "EUR_GBP", "CHF_GBP", "JPY_GBP", "CNY_GBP")
               cbind(CAD_GBP, USD_GBP, EUR_GBP, CHF_GBP, JPY_GBP, CNY_GBP)
           } else if (grepl("ZCB", type.series) & grepl("USD", type.series)) {
               data("ZCB_USD")
               ZCB_USD / 100
           } else if (grepl("ZCB", type.series) & grepl("CAD", type.series)) {
               data("ZCB_CAD")
               ZCB_CAD / 100
           } else {
               stop ("Wrong 'type.series'")
           }

    ## Extract the original (price) time series for both training and test period
    pseries.train <- as.matrix(raw[paste0(train.period, collapse = "/")])
    pseries.test  <- as.matrix(raw[paste0(test.period,  collapse = "/")])

    ## Convert to risk-factor changes for both training and test periods
    rseries.train <- returns(pseries.train, ...)
    ## Note the first return in the test period is based on the change in price
    ## from the last day of the training period
    rseries.test <- returns(rbind(tail(pseries.train, n = 1), pseries.test), ...)
    list(train = rseries.train, test = rseries.test)
}


### 0.2 Modeling marginal time series using ARMA-GARCH #########################

##' @title Marginal Time Series Modeling using ARMA-GARCH
##' @param data matrix containing the time series data (i.e. training data)
##' @param file character string with ending .rds containing the file name used
##'        to save the fitted models
##' @param garch.order 2-integer containing the GARCH orders
##' @param arma.order 2-integer containing the ARMA orders
##' @param innov.model character string containing the choice of innovation
##'        distribution (by default t)
##' @param with.mu logical indicating if we include the 'mu' mean parameter in
##'        the ARMA model
##' @return list containing fitted ARMA-GARCH models
marginal_ts_fit <- function(data, file, garch.order = c(1,1), arma.order = c(1,1),
                            innov.model = "std", with.mu = TRUE)
{
    if (file.exists(file)) {
        fitted.models <- readRDS(file)
    } else {
        ## Specify the marginal ARMA-GARCH models
        spec <- rep(list(ugarchspec(variance.model = list(model = "sGARCH",
                                                          garchOrder = garch.order),
                                    mean.model = list(armaOrder = arma.order,
                                                      include.mean = with.mu),
                                    distribution.model = innov.model)),
                    ncol(data))
        fitted.models <- fit_ARMA_GARCH(data, ugarchspec.list = spec, solver = 'hybrid')
        saveRDS(fitted.models, file = file)
    }
    fitted.models
}


### 0.3 Modeling cross-sectional dependence ####################################

##' @title Modeling Dependence using Copulas and GMMNs
##' @param U matrix of pseudo-observations of the training data
##' @param GMMN.dim numeric vector of length at least two giving the dimensions
##'        of the input layer, the hidden layer(s) (if any) and the output layer;
##'        only needed if fitting GMMNs.
##' @param file character string with ending .rda
##' @return fitted copula or GMMN models
dependence_fit <- function(U,GMMN.dim, file)
{
    if (exists_rda(file, names = rm_ext(basename(file)), package = package)) {
        fitted.model <- read_rda(file = file, names = rm_ext(basename(file)))
        if(grepl("GMMN",file)) fitted.model <- to_callable(fitted.model) # unserialization for GMMNs
    } else { # fitting and saving copulas and GMMNs
        dep.types <- c("norm_ex", "t_ex", "t_un", "gumbel", "GMMN") # types of models
        dm <- dim(U)
        n <- dm[1]
        d <- dm[2]
        ind <- which(sapply(dep.types, function(x) grepl(x, file))) # index of dep.types list
        stopifnot(length(ind) == 1) # check that there is only one match
        fitted.model <-
            switch(dep.types[ind],
                   "norm_ex" = {
                       fitCopula(normalCopula(dim = d),
                                 data = U, method = "mpl", estimate.variance = FALSE)
                   },
                   "t_ex" = {
                       fitCopula(tCopula(dim = d),
                                 data = U, method = "mpl", estimate.variance = FALSE)
                   },
                   "t_un" = {
                       fitCopula(tCopula(dim = d, dispstr = "un"),
                                 data = U, method = "mpl", estimate.variance = FALSE)
                   },
                   "gumbel" = {
                       fitCopula(gumbelCopula(dim = d),
                                 data = U, method = "mpl", estimate.variance = FALSE)
                   },
                   "GMMN" = {
                       train_once(GMMN_model(GMMN.dim), data = U,
                                  batch.size = n, nepoch = nepo, file = file,
                                  package = package)
                   },
                   stop("Wrong 'method'"))
        ## Since train_once() already saves GMMN models we only need to save the copula models
        if(!grepl("GMMN",file)) save_rda(fitted.model, file = file)
    }
    fitted.model
}


### 0.4 Modeling multivariate time series ######################################

##' @title Fitting all Dependence Models for a Specified Dataset
##' @param type.series see get.ts()
##' @param train.period see get.ts()
##' @param test.period see get.ts()
##' @param with.mu see marginal_ts_fit()
##' @param pca.dim numeric value specifying the number of PCs to be used for
##'        dimension reduction
##' @return list containing the fitted marginal models, the PCA model (if used)
##'         and a list of fitted dependence models (4 copulas and 3 GMMNs)
all_multivariate_ts_fit <- function(type.series, train.period, test.period,
                                    pca.dim = NULL, with.mu = TRUE)
{
    ## For interest rate data we have a handful of different specifications
    if (grepl("ZCB", type.series)) {
        X <- get_ts(type.series = type.series, train.period = train.period,
                    test.period = test.period, method = "diff")
        with.mu <- FALSE
    } else {
        X <- get_ts(type.series = type.series, train.period = train.period,
                    test.period = test.period)
    }

    ## File name for loading-saving ARMA-GARCH associated with type.series
    marginal.file <- paste0("ARMA_GARCH_", paste0(train.period, collapse = "_"),
                            "_", type.series, ".rds")

    ## 1) First we feed the training dataset to marginal_ts_fit()
    marginal.models <- marginal_ts_fit(data = X$train, with.mu = with.mu,
                                       file = marginal.file) # fit marginal time series models
    standard.resid <- lapply(marginal.models$fit, residuals, standardize = TRUE) # grab out standardized residuals
    Y <- as.matrix(do.call(merge, standard.resid)) # converting residuals to matrix data

    ## 2) If dimension reduction is required, apply PCA
    if(!is.null(pca.dim)) {
        PCA.model <- PCA_trafo(Y)
        Y <- PCA.model$PCs[,1:pca.dim]
    } else {
        PCA.model <- NULL
    }

    ## 3) Obtain pseudo-observations
    U <- pobs(Y)
    dimnames(U) <- NULL

    ## Setup
    dm <- dim(U)
    ntrn <- dm[1] # number of observations in the training dataset
    dim.in.out <- dm[2] # dimension of pseudo-observations used to train the dependence models
    dim.hid1 <- 100 # GMMN hyperparameters
    dim.hid2 <- 300
    dim.hid3 <- 600
    nbat <- ntrn

    ## 4) Fitting the four copula models
    file.gumbel  <- paste0("cop_","gumbel", "_dim_",dim.in.out,
                           if(!is.null(pca.dim)) "_PCA","_",type.series,".rda")
    file.norm.ex <- paste0("cop_","norm_ex","_dim_",dim.in.out,
                           if(!is.null(pca.dim)) "_PCA","_",type.series,".rda")
    file.t.ex    <- paste0("cop_","t_ex",   "_dim_",dim.in.out,
                           if(!is.null(pca.dim)) "_PCA","_",type.series,".rda")
    file.t.un    <- paste0("cop_","t_un",   "_dim_",dim.in.out,
                           if(!is.null(pca.dim)) "_PCA","_",type.series,".rda")
    model.gumbel  <- dependence_fit(U = U, file = file.gumbel)
    model.norm.ex <- dependence_fit(U = U, file = file.norm.ex)
    model.t.ex    <- dependence_fit(U = U, file = file.t.ex)
    model.t.un    <- dependence_fit(U = U, file = file.t.un)

    ## 5) Fitting the three GMMN models
    file.G1 <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid1,"_",dim.in.out,"_ntrn_",ntrn,"_nbat_",nbat,
                      "_nepo_",nepo, if(!is.null(pca.dim)) "_PCA","_",type.series,".rda")
    file.G2 <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid2,"_",dim.in.out,"_ntrn_",ntrn,"_nbat_",nbat,
                      "_nepo_",nepo, if(!is.null(pca.dim)) "_PCA","_",type.series,".rda")
    file.G3 <- paste0("GMMN_dim_",dim.in.out,"_",dim.hid3,"_",dim.in.out,"_ntrn_",ntrn,"_nbat_",nbat,
                      "_nepo_",nepo, if(!is.null(pca.dim)) "_PCA","_",type.series,".rda")
    model.G1 <- dependence_fit(U = U, GMMN.dim = c(dim.in.out, dim.hid1, dim.in.out), file = file.G1)
    model.G2 <- dependence_fit(U = U, GMMN.dim = c(dim.in.out, dim.hid2, dim.in.out), file = file.G2)
    model.G3 <- dependence_fit(U = U, GMMN.dim = c(dim.in.out, dim.hid3, dim.in.out), file = file.G3)

    ## 6) Results
    dependence.models <- list(model.indep   = NULL,
                              model.gumbel  = model.gumbel,
                              model.norm.ex = model.norm.ex,
                              model.t.ex    = model.t.ex,
                              model.t.un    = model.t.un,
                              model.G1      = model.G1,
                              model.G2      = model.G2,
                              model.G3      = model.G3)
    list(marginal = marginal.models, PCA = PCA.model, dependence = dependence.models) # return
}


### 0.5 Evaluating fitted dependence models using Maximum Mean Discrepancy (MMD)

##' @title Extract the Dependence of Multivariate Time Series in the Test Period
##' @param type.series see get.ts()
##' @param train.period see get.ts()
##' @param test.period see get.ts()
##' @param with.mu see marginal_ts_fit()
##' @param pca.dim see all_multivariate_ts_fit()
##' @return (tau, d*)-matrix containing the underlying dependence of the test dataset
extract_dependence_ts <- function(type.series,train.period,test.period,pca.dim=NULL,with.mu=TRUE)
{
    ## For interest rate data we have a handful of different specifications
    if (grepl("ZCB", type.series)) {
        X <- get_ts(type.series = type.series, train.period = train.period,
                    test.period = test.period, method = "diff")
        with.mu <- FALSE
    } else {
        X <- get_ts(type.series = type.series, train.period = train.period,
                    test.period = test.period)
    }

    ## Setup
    dm <- dim(X$test)
    n <- dm[1] # number of observations in test data
    d <- dm[2] # dimension of time series data

    ## File name for loading-saving ARMA-GARCH associated with type.series
    marginal.file <- paste0("ARMA_GARCH_", paste0(train.period, collapse = "_"),
                            "_", type.series, ".rds")

    ## 1) First we feed testing dataset to marginal_ts_fit()
    marginal.fits <- marginal_ts_fit(data = X$train, with.mu = with.mu,
                                     file = marginal.file)$fit # fit marginal time series models
    standard.resid <- lapply(marginal.fits, residuals, standardize = TRUE) # grab out standardized residuals from
    Y <- as.matrix(do.call(merge, standard.resid)) # converting residuals to matrix data
    dimnames(Y) <- NULL # remove any dimension names to not interfere with functions like predict()

    ## 2) If dimension reduction is required, apply PCA
    if(!is.null(pca.dim)) PCA <- prcomp(Y)

    ## Retrieve fitted marginal ARMA-GARCH model specifications
    specs <- sapply(1:d,function(i) getspec(marginal.fits[[i]]))
    for (i in 1:d)
        setfixed(specs[[i]]) <- coef(marginal.fits[[i]])

    ## 4) Use fitted ARMA-GARCH model to extract model-implied standardized residuals
    ##    on the test data. To achieve this we make use of a rolling forecast to obtain
    ##    the fitted model implied mean and sigma process into the future (test data).
    test.mean <- sapply(1:d, function (i)
        fitted(ugarchforecast(specs[[i]], data = X$test[,i], n.ahead = 1,
                              n.roll = n-1, out.sample = n-1)))
    test.sigma <- sapply(1:d, function (i)
        sigma(ugarchforecast(specs[[i]], data = X$test[,i], n.ahead = 1,
                             n.roll = n-1, out.sample = n-1)))
    Y.test <- (X$test - test.mean) / test.sigma

    ## If we include dimension reduction, we make use of the PCA model on the training
    ## data to project the standardized residuals of the test data onto the lower-dim-
    ## ensional PC space.
    if (!is.null(pca.dim)) Y.test <- predict(PCA, Y.test)[,1:pca.dim]

    ## Return
    pobs(Y.test)
}

##' @title MMD between Generated Samples from the Dependence Models (Copulas, GMMNs)
##'        and the Empirical Dependence of the Test Dataset
##' @param generators list of generators from 8 dependence models
##'        (independence, 4 copulas, 3 GMMNs)
##' @param U.test matrix containing the empirical dependence structure of the multivariate
##'        time series in the test period
##' @param bandwidth numeric (vector) containing the bandwidth parameters (sigma).
##' @return numeric vector of length 8 containing MMD values for the 8 models.
MMD_eval <- function(generators, U.test, bandwidth = seq(0.1, 0.9, by = 0.2))
{
    ## Setup
    dm <- dim(U.test)
    n <- dm[1] # test dataset sample size
    d <- dm[2] # test dataset dimension
    U.test <- tf$constant(U.test, dtype = tf$float32) # conversion to TensorFlow object

    ## Generate samples from independence, fitted parametric copulas or fitted GMMNs
    U.models <- lapply(1:length(generators), function(i)
        if (grepl("indep", names(generators)[i])) {
            tf$constant(matrix(runif(n * d), ncol = d), dtype = tf$float32)
        }  else if (grepl("cop", names(generators)[i])) {
            tf$constant(rCopula(n, copula = generators[[i]]), dtype = tf$float32)
        }  else if (grepl("G", names(generators)[[i]])) {
            tf$constant(pobs(predict(generators[[i]], x = matrix(rnorm(n * d), ncol = d))),
                        dtype = tf$float32)
        })

    ## Compute and return MMD values
    sapply(1:length(U.models), function(i)
        if(!tf$executing_eagerly()) {
            sess$run(  loss(U.models[[i]], U.test, type = "MMD", bandwidth = bandwidth))
        } else {
            as.numeric(loss(U.models[[i]], U.test, type = "MMD", bandwidth = bandwidth))
        })
}

##' @title B Realizations of the MMD Metric
##' @param type.series see get.ts()
##' @param train.period see get.ts()
##' @param test.period see get.ts()
##' @param with.mu see marginal_ts_fit()
##' @param pca.dim see all_multivariate_ts_fit()
##' @param B numeric value specifying the number of realization of the MMD
##'        statistic to compute
##' @return (8,B) matrix containing B realizations for 8 models
##'         (independence, 4 copulas, 3 GMMNs)
##' @note When using the function ensure sess <- tf$Session() is specified in the
##'       global environment. Once all such computation is finished use sess$close()
MMD_metric <- function(type.series, train.period, test.period, with.mu = TRUE,
                       pca.dim = NULL, B = 100)
{
    ## Obtain the empirical dependence of the test data
    U.test <- extract_dependence_ts(type.series = type.series, train.period = train.period,
                                    test.period = test.period, with.mu = with.mu, pca.dim = pca.dim)
    dm <- dim(U.test)
    d <- dm[2] # dimension of the empirical dependence structure of the test data
    file <- paste0("MMD","_dim_",d,"_B_",B,"_",type.series,".rds")
    if(file.exists(file)) {
        mmd.vals <- readRDS(file)
    } else {
        models <- all_multivariate_ts_fit(type.series = type.series, train.period = train.period,
                                          test.period = test.period, with.mu = with.mu, pca.dim = pca.dim)
        ## List of dependence model generators
        generators <- list(gen.indep      = models$dependence$model.indep,
                           gen.copgumbel  = models$dependence$model.gumbel@copula,
                           gen.copnorm.ex = models$dependence$model.norm.ex@copula,
                           gen.copt.ex    = models$dependence$model.t.ex@copula,
                           gen.copt.un    = models$dependence$model.t.un@copula,
                           gen.G1         = models$dependence$model.G1[["model"]],
                           gen.G2         = models$dependence$model.G2[["model"]],
                           gen.G3         = models$dependence$model.G3[["model"]])
        set.seed(271) # for reproducibility
        mmd.vals <- replicate(B, MMD_eval(generators = generators, U.test = U.test))
        saveRDS(mmd.vals, file = file)
    }
    mmd.vals
}


### 0.6 Forecasting the one-period ahead empirical predictive distribution #####

##' @title (Rolling) Empirical Distribution Forecasts with Training Period Fixed
##'        (i.e. models are not re-fitted)
##' @param n.samples numeric value specifying number of samples (or paths) used
##'        to construct the empirical distribution forecasts with
##' @param n.test numeric value indicating the number of time periods in the
##'        test/evaluation period.
##' @param margin.model list containing fitted ARMA-GARCH time series models
##' @param dep.model fitted dependence model generator (copula or GMMN; NULL
##'        in the independence case)
##' @param PCA.model (fitted) PCA model; return value of PCA_trafo()
##' @param initial.ts (n.test, d)-dimensional matrix containing initial time
##'        series values used for the (rolling) simulation-based forecasts.
##' @param h numeric value signifying the number of periods ahead for the
##'        distribution forecast (defaults to 1).
##' @param dep.model character string with the three choices indicating the
##'        type of dependence model ("indep", "copula" or "GMMN").
##' @param pca.dim numeric specifying the number of PCs used for dimension
##'        reduction.
##' @return list of length n.test with each element of the list containing
##'         a (h * n.samples, d)-matrix providing the h-day ahead
##'         empirical distribution forecast.
distribution_forecast_ts <- function(n.samples, n.test, margin.model, dep.model, PCA.model,
                                     initial.ts, pca.dim = NULL, h = 1,
                                     type.dep = c("GMMN", "copula", "indep"))
{
    ## Setup
    type <- match.arg(type.dep)
    marginal.fits <- margin.model$fit # fitted marginal models
    nus.fits <- sapply(marginal.fits, function(x) x@fit$coef[["shape"]]) # fitted d.o.f.
    if (!is.null(pca.dim)) { # grab the required components of the PCA model to be used later
        Y.pc <- PCA.model$PCs[,1:pca.dim]
        mu.pc <- PCA.model$mu
        Gamma.pc <- PCA.model$Gamma
    }
    d <- ncol(initial.ts) # dimension of the multivariate time series data
    dim.dep <- if (!is.null(pca.dim)) pca.dim else dim.dep <- d # dimension of the dependence model

    ## Start looping through test period
    returns.dist <- vector("list", length = n.test)
    for (i in 1:n.test)
    {
        initial.returns <- initial.ts[i,]
        if (i==1) {
            ## For the first period (time step), we set the initial conditional
            ## sigma and residuals using the fitted (on the training dataset)
            ## marginal models
            initial.sigma    <- sapply(1:d, function (j) tail(marginal.fits[[j]]@fit$sigma, n = 1))
            initial.residual <- sapply(1:d, function (j) tail(marginal.fits[[j]]@fit$residuals, n = 1))
        } else {
            ## sigmaSim found in the temp.sim object (from the previous time point i-1)
            ## gives us an (h, n.sample)-matrix. As we re-compute a new distributional
            ## forecast in each time point (day) we just need the first row of the given
            ## matrix. Moreover the first row of this matrix will always be a constant
            ## across all n.sample values. Thus we can just choose the first value.
            ## This is naturally repeated across all d dimensions. This also generalizes
            ## to h > 1, since even in that case we may want to generate a new h-period
            ## ahead distributional forecast starting from each point i in the
            ## test/evaluation period. That is, this is suited for a sort of rolling
            ## h-day ahead forecast that is re-computed at the end of each period for
            ## the next h-periods. In that case, taking the first value of this matrix
            ## suffices as the first row will also be constant even when temp.sim object
            ## is for h > 1. Thus this d-dimensional conditional sigma is used as an
            ## initialization for simulating the predictive distribution for the next day.
            initial.sigma <- sapply(1:d, function(j) temp.sim[[j]]@simulation$sigmaSim[1])
            ## To obtain the initial residual at time point i that is based on the
            ## "realized" return values in the test point, we need to go about this
            ## in a round about way. We first calculate the conditional mean at time
            ## point i-1 using series and residual simulation from the temp.sim object.
            ## As before this gives an (h, n.sample)-matrix. The difference between
            ## the seriesSim and residSim is a constant across n.sample values in the
            ## first row. As a result, it suffices to use the first value. This is then
            ## repeated across all d dimensions.
            initial.mean <- sapply(1:d, function(j) (temp.sim[[j]]@simulation$seriesSim - temp.sim[[j]]@simulation$residSim)[1])
            initial.residual <- initial.returns - initial.mean
        }
        ## Generating samples from the various dependence models
        U.sim <- switch(type,
               "GMMN" = {
                   N01.prior.gen <- matrix(rnorm(n.samples * dim.dep * h), ncol = dim.dep)
                   pobs(predict(dep.model, x = N01.prior.gen))
               },
               "copula" = {
                   rCopula(n.samples * h, copula = dep.model)
               },
               "indep" = {
                   matrix(runif(n.samples * dim.dep * h), ncol = dim.dep)
               },
               stop("Wrong 'type'"))
        if (!is.null(pca.dim)) {
            ## When we applied PCA, we empirically modeled the marginals distributions
            ## of the top pca.dim PCs, so we need to first add back the empirical margins
            ## to our samples from the dependence model.
            Y.pc.sim <- toEmpMargins(U = U.sim, x = Y.pc)
            Z.sim <- PCA_trafo(x = Y.pc.sim, mu = mu.pc, Gamma = Gamma.pc, inverse = TRUE)
        } else {
            ## When we do not apply PCA, we simply add back the (scaled) t marginal
            ## distribution as specified as part of our ARMA-GARCH model
            Z.sim <- sapply(1:d, function(j)
                sqrt((nus.fits[j]-2)/nus.fits[j]) * qt(U.sim[,j], df = nus.fits[j]))
        }
        temp.sim <- lapply(1:d, function(j)
            ugarchsim(marginal.fits[[j]],
                      n.sim = h,
                      m.sim = n.samples,
                      startMethod = 'sample',
                      presigma     = initial.sigma[j], # Feed in the initial conditional sigmas here
                      prereturns   = initial.returns[j], # Feed in the initial returns here
                      preresiduals = initial.residual[j], # Feed in the initial residuals here
                      custom.dist = list(name = "sample", # our innovations
                                         distfit = matrix(Z.sim[,j], ncol = n.samples))))
        ## Grab out and record the return distributions at each time point
        returns.dist[[i]] <- sapply(temp.sim, function(x) fitted(x))
    }
    ## Return a list containing n.test-many empirical predictive distributions,
    ## each based on n.samples
    returns.dist
}

##' @title Empirical Distribution Forecasts for all Fitted Multivariate Time Series Models
##' @param type.series see get.ts()
##' @param train.period see get.ts()
##' @param test.period see get.ts()
##' @param with.mu see marginal_ts_fit()
##' @param pca.dim see all_multivariate_ts_fit()
##' @param n.samples see distribution_forecast_ts()
##' @param h see distribution_forecast_ts()
##' @return 8-list corresponding to the 8 dependence models considered
##'         (3 GMMNs, 4 copulas and independence); each element is
##'         a list of type as returned by distribution_forecast_ts().
##' @note Intermediate results are not saved due to size.
all_distribution_forecast_ts <- function(type.series, train.period, test.period, with.mu = TRUE,
                                         pca.dim = NULL, n.samples = 1000, h = 1)
{
    ## For interest rate data we have a handful of different specifications
    if (grepl("ZCB", type.series)) {
        X <- get_ts(type.series = type.series, train.period = train.period,
                    test.period = test.period, method = "diff")
        with.mu <- FALSE
    } else {
        X <- get_ts(type.series = type.series, train.period = train.period,
                    test.period = test.period)
    }

    ## Grab out models
    n.test <- nrow(X$test) # number of observation in the test dataset
    models <- all_multivariate_ts_fit(type.series  = type.series, # all fitted multiv. time series models
                                      train.period = train.period,
                                      test.period  = test.period,
                                      with.mu = with.mu, pca.dim = pca.dim)
    margin.models <- models$marginal # marginal models
    PCA.model     <- models$PCA # PCA
    dep.models    <- models$dependence # dependence models

    ## To initialize the matrix of return time series for the recursive
    ## forecasting procedure we need to attach the last return from the
    ## training dataset (initial value for our first period prediction)
    ## and remove the last return in the test dataset (will be the
    ## initial value for the time point outside of the test period).
    initial.ts<-rbind(as.numeric(tail(X$train,n=1)),X$test[-c(n.test),])
    set.seed(271) # set seed for reproducibility
    distr.pred.models <- lapply(1:length(dep.models), function(i) {
        ## Determine type of dependence
        type.dep <- if (grepl("indep", names(dep.models)[i])) {
            "indep"
        } else if(grepl("G", names(dep.models)[i])) {
            "GMMN"
        } else "copula"
        distribution_forecast_ts(n.samples = n.samples, n.test = n.test,
                                 margin.model = margin.models,
                                 dep.model = if(type.dep == "indep") {
                                                 dep.models[[i]]
                                             } else if(type.dep == "copula") {
                                                 dep.models[[i]]@copula
                                             } else if (type.dep=="GMMN") dep.models[[i]][["model"]],
                                 PCA.model = PCA.model,
                                 initial.ts = initial.ts, pca.dim = pca.dim, h = h,
                                 type.dep = type.dep)
    })
    names(distr.pred.models) = c("distr.pred.indep",
                                 "distr.pred.gumbel", "distr.pred.norm.ex", "distr.pred.t.ex", "distr.pred.t.un",
                                 "distr.pred.G1", "distr.pred.G2", "distr.pred.G3")

    ## Return
    distr.pred.models
}


### 0.7 Evaluating quality of distribution forecasts ###########################

##' @title Multivariate Times Series Distribution Forecasts based on Metrics
##' @param distribution.forecasts list of distribution forecasts constructed
##'        from fixed ARMA-GARCH and (if used) PCA models and various dependence
##'        models. A list of type as returned by all_distribution_forecast_ts().
##' @param X list containing the training and test data for a multivariate time series
##' @param type.metric forecast evaluation metrics including the variogram score,
##'        MSE and absolute error of portfolio VaR exceedances, i.e.
##'        abs(actual exceedances - expected exceedances)
##' @param file character string specifying the file name used to save the
##'        forecast evaluation metrics
##' @param alpha significance level of portfolio VaR exceedance (here:
##'        lower quantile, small value)
##' @param p numeric value specifying the order of the variogram score
##' @return 8-list (one component for each dependence model considered).
##'         For each element of this list the function returns a numeric
##'         vector of length of the test period representing the forecast
##'         evaluation metric for each one-period ahead forecast.
distribution_forecast_eval <- function(distribution.forecasts, X,
                                       type.metric = c("variogram_score", "MSE",
                                                       "VaR_exceed_abs_error"),
                                       file, alpha = 0.05, p = 0.5)
{
    if (file.exists(file)) {
        metrics <- readRDS(file)
    } else {
        type <- match.arg(type.metric)
        X.test <- X$test # grab out test dataset
        dm <- dim(X.test)
        n.test <- dm[1] # number of observations in test set
        d.test <- dm[2] # dimension of test dataset
        n.samples <- nrow(distribution.forecasts[[1]][[1]]) # number of samples to construct the empirical distribution forecast
        switch(type,
               "variogram_score" = { # scoringRules::vs_sample (p = 0.5 by default)
                   metrics <- lapply(1:length(distribution.forecasts), function(j)
                       sapply(1:n.test, function(i)
                           vs_sample(y = X.test[i,], dat = t(distribution.forecasts[[j]][[i]]), p = p)))
               },
               "MSE" = { # (multivariate) version of MSE calculated using Euclidean distances
                   metrics <- lapply(1:length(distribution.forecasts), function(k)
                       sapply(1:n.test, function(i)
                           mean(sapply(1:n.samples, function(j)
                               dist(rbind(distribution.forecasts[[k]][[i]][j,], X.test[i,]))))))
               },
               "VaR_exceed_abs_error" = { # portfolio VaR exceedance evaluation metric
                   ## Aggregate returns
                   sum.test <- rowSums(X.test) # for the test data set
                   sum.forecasts <- lapply(1:length(distribution.forecasts), function(j) # for each path and dependence model
                       sapply(1:n.test, function(i) rowSums(distribution.forecasts[[j]][[i]])))

                   ## Approximate the alpha-quantile based on the distribution forecasts
                   ## of the aggregate sum
                   sumquantile.forecasts <- lapply(1:length(distribution.forecasts), function(j)
                       sapply(1:n.test, function(i) as.numeric(quantile(sum.forecasts[[j]][,i], probs = alpha))))

                   ## Calculate the number of exceedances in the test data at the
                   ## various quantile levels implied by each of our distribution forecasts
                   exceedance.forecasts <- lapply(1:length(distribution.forecasts), function(j)
                       sum(sum.test < sumquantile.forecasts[[j]]) / n.test)

                   ## Simple absolute error between the expected (alpha) and the actual exceedances
                   metrics <- lapply(1:length(distribution.forecasts), function(j) abs(exceedance.forecasts[[j]]-alpha))
               },
               stop("Wrong 'type'"))
        names(metrics) <- names(distribution.forecasts) # preserve model names for output
        saveRDS(metrics, file = file)
    }
    metrics
}


### 0.8 Plot ###################################################################

##' @title Scatter plot of MMD metrics vs forecast metrics
##' @param type.series see get.ts()
##' @param train.period see get.ts()
##' @param test.period see get.ts()
##' @param type.metric forecast evaluation metric: variogram score, MSE and
##'        absolute error of portfolio VaR exceedance, i.e.
##'        abs(actual exceedance - expected exceedance)
##' @param alpha The alpha used for calculating the portfolio VaR exceedance
##' @param with.mu see marginal_ts_fit()
##' @param pca.dim see all_multivariate_ts_fit()
##' @param n.samples see distribution_forecast_ts()
##' @param B see MMD_metric()
##' @param p see distribution_forecast_eval()
##' @return invisible(); produces a plot as side-effect.
forecast_evaluation_plot <- function(type.series, train.period, test.period,
                                     type.metric, alpha = 0.05, with.mu = TRUE,
                                     pca.dim = NULL, n.samples = 1000, B = 100, p = 0.5)
{
    ## For interest rate data we have a handful of different specifications
    if (grepl("ZCB", type.series)) {
        X <- get_ts(type.series = type.series, train.period = train.period,
                    test.period = test.period, method = "diff")
        with.mu <- FALSE
    } else {
        X <- get_ts(type.series = type.series, train.period = train.period,
                    test.period = test.period)
    }

    ## MMD metrics
    MMD.metrics <- MMD_metric(type.series = type.series, train.period = train.period,
                              test.period = test.period, with.mu = with.mu, pca.dim = pca.dim,
                              B = B)

    ## File name for saving/loading forecast metrics
    dep.dim <- if (!is.null(pca.dim)) pca.dim else ncol(X$test) # dependence dimension (for file name)
    filename.metrics <- paste0(type.metric, if(grepl("VaR_exceed_abs_error",type.metric)) "_alpha_",
                               if(grepl("VaR_exceed_abs_error",type.metric)) alpha,
                               if(grepl("variogram_score",type.metric)) "_p_",
                               if(grepl("variogram_score",type.metric)) p,
                               "_depdim_",dep.dim,"_",type.series,".rds")

    ## One-period ahead distribution forecast and its evaluation based on type.metric
    if (file.exists(filename.metrics)) {
        forecast.metrics <- readRDS(filename.metrics)
    } else {
        distribution.forecasts <- all_distribution_forecast_ts(type.series = type.series,
                                                               train.period = train.period,
                                                               test.period = test.period,
                                                               with.mu = with.mu, pca.dim = pca.dim,
                                                               n.samples = n.samples)
        forecast.metrics <- distribution_forecast_eval(distribution.forecasts = distribution.forecasts,
                                                       X = X, type.metric = type.metric,
                                                       alpha = alpha, file = filename.metrics, p = p)
    }

    ## Average MMD and forecast metric across the test data set
    average.MMD.metrics <- rowMeans(MMD.metrics)
    average.forecast.metrics <- sapply(1:length(forecast.metrics), function(i)
        mean(forecast.metrics[[i]]))
    names(average.forecast.metrics) <- names(forecast.metrics) # preserve model name

    ## Create a vector of labels with each label corresponding to a dependence model
    labels.vec <- rep(NA, length(forecast.metrics))
    labels.vec[which(grepl("indep",  names(average.forecast.metrics)))] <- "Independent"
    labels.vec[which(grepl("gumbel", names(average.forecast.metrics)))] <- "Gumbel copula"
    labels.vec[which(grepl("norm.ex",names(average.forecast.metrics)))] <- "Normal copula (exchangeable)"
    labels.vec[which(grepl("t.ex",   names(average.forecast.metrics)))] <- "t copula (exchangeable)"
    labels.vec[which(grepl("t.un",   names(average.forecast.metrics)))] <- "t copula (unstructured)"
    labels.vec[which(grepl("G1",     names(average.forecast.metrics)))] <- "GMMN model 1"
    labels.vec[which(grepl("G2",     names(average.forecast.metrics)))] <- "GMMN model 2"
    labels.vec[which(grepl("G3",     names(average.forecast.metrics)))] <- "GMMN model 3"
    cols.vec <- c(rep(cols[4], 5), rep(cols[7], 3))
    pch.vec <- 1:length(average.forecast.metrics)

    ## Create labels for y-axis depending on the type of forecast evaluation metric.
    ylabel <- if(grepl("MSE",type.metric)) {
        "AMSE"
    } else if (grepl("variogram_score", type.metric)) {
        substitute("AVS"^p., list(p. = p))
    } else if (grepl("VaR_exceed_abs_error", type.metric)) {
        substitute("VEAR"[alpha.], list(alpha. = alpha))
    }

    ## Plotting average MMD vs average forecast evaluation metric
    filename.plot <- paste0("fig_MMD","_vs_",rm_ext(filename.metrics),".pdf")
    doPDF <- require(crop)
    if(doPDF) pdf(file = filename.plot)
    plot(average.MMD.metrics, average.forecast.metrics, xlab = 'AMMD',
         ylab = ylabel, pch = pch.vec, col = cols.vec)
    legend(x = "topleft", cex = 0.6, legend = labels.vec,
           bty = 'n', pch = pch.vec, col = cols.vec)
    if(doPDF) dev.off.crop(filename.plot)
}


### 1 Computing all results ####################################################

if(!tf$executing_eagerly())
    sess <- tf$Session() # called globally for the MMD calculations


### 1.1 Plots for US exchange rate data ########################################

## Results where MMD is computed using fitted ARMA-GARCH models and PCA
## models projected onto the test dataset

train.period1 <- c("2000-01-01", "2014-12-31")
train.period2 <- c("1995-01-01", "2014-12-31")
test.period   <- c("2015-01-01", "2015-12-31")

## Results for all MSE, variogram score and VaR exceedance absolute error
## (with alpha = 0.05) evaluation metrics
human_time(forecast_evaluation_plot(type.series = "FX_USD", # ~= 34min
                                    train.period = train.period1,
                                    test.period = test.period,
                                    type.metric = "MSE"))
human_time(forecast_evaluation_plot(type.series = "FX_USD", # ~= 3min
                                    train.period = train.period1,
                                    test.period = test.period,
                                    type.metric = "variogram_score",
                                    p = 0.25))
human_time(forecast_evaluation_plot(type.series = "FX_USD", # ~= 3min
                                    train.period = train.period1,
                                    test.period = test.period,
                                    type.metric = "VaR_exceed_abs_error"))


### 1.2 Plots for GBP exchange rate data #######################################

## Results where MMD is computed using fitted ARMA-GARCH models and PCA
## models projected onto the test dataset

## Results for all MSE, variogram score and VaR exceedance absolute error
## (with alpha = 0.05) evaluation metrics
human_time(forecast_evaluation_plot(type.series = "FX_GBP", # ~= 1.2h
                                    train.period = train.period1,
                                    test.period = test.period,
                                    type.metric = "MSE"))
human_time(forecast_evaluation_plot(type.series = "FX_GBP", # ~= 5min
                                    train.period = train.period1,
                                    test.period = test.period,
                                    type.metric = "variogram_score",
                                    p = 0.25))
human_time(forecast_evaluation_plot(type.series = "FX_GBP", # ~= 4min
                                    train.period = train.period1,
                                    test.period = test.period,
                                    type.metric = "VaR_exceed_abs_error"))


### 1.3 Plots for US interest rate data ########################################

## For this data we fix the PCA dimension to be 3 (originally 30).

## Results where MMD is computed using fitted ARMA-GARCH models and PCA
## models projected onto the test dataset

## Obtain results for only MSE, variogram score evaluation metrics

## pca.dim = 3
human_time(forecast_evaluation_plot(type.series = "ZCB_USD", # ~= 1.8h
                                    train.period = train.period2,
                                    test.period = test.period,
                                    type.metric = "MSE",
                                    pca.dim = 3))
human_time(forecast_evaluation_plot(type.series = "ZCB_USD", # ~= 8min
                                    train.period = train.period2,
                                    test.period = test.period,
                                    type.metric = "variogram_score", p = 0.25,
                                    pca.dim = 3))


### 1.4 Plots for US interest rate data ########################################

## For this data we experiment with PCA dimension equal to 4 (originally 120).

## Results where MMD is computed using fitted ARMA-GARCH models and PCA
## models projected onto the test dataset

## pca.dim = 4
human_time(forecast_evaluation_plot(type.series = "ZCB_CAD", # ~= 2.4h
                                    train.period = train.period2,
                                    test.period = test.period,
                                    type.metric = "MSE",
                                    pca.dim = 4))
human_time(forecast_evaluation_plot(type.series = "ZCB_CAD", # ~= 44min
                                    train.period = train.period2,
                                    test.period = test.period,
                                    type.metric = "variogram_score", p = 0.25,
                                    pca.dim = 4))

if(!tf$executing_eagerly()) sess$close()


sessionInfo()