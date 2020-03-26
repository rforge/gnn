## By Marius Hofert and Avinash Prasad

## Data example for Hofert, Prasad, Zhu ("Quasi-random sampling for multivariate
## distributions via generative neural networks"). The NNs were trained on an
## NVIDIA Tesla P100 GPU.


### Setup ######################################################################

## Packages
library(keras) # interface to Keras (high-level neural network API)
library(tensorflow) # interface to TensorFlow (numerical computation with tensors)
## => would allow to set the seed with use_session_with_seed(271), but then no GPU or CPU parallelism
if(grepl("gra", Sys.info()[["nodename"]])) {
    tf_version() # dummy call to activate connection to TensorFlow (any first call will fail on the cluster; here: NULL)
    use_virtualenv(Sys.getenv('VIRTUAL_ENV')) # tensorflow command to access the activated Python environment
}
library(qrng) # for sobol()
if(packageVersion("copula") < "0.999.19")
    stop('Consider updating via install.packages("copula", repos = "http://R-Forge.R-project.org")')
library(copula) # for the considered copulas
library(gnn) # for the used GMMN models
library(xts) # for na.fill()
library(rugarch)
library(MASS)
library(qrmtools) # for ES_t(), ARMA_GARCH_fit()
library(qrmdata) # for required datasets

## Global training parameters
dim.hid <- 300L # dimension of the (single) hidden layer
nepo <- 300L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
ngen <- 10000L # sample size of the generated data

## Plots
doPDF <- require(crop) # crop if 'crop' is available


### 0 Auxiliary functions ######################################################

### 0.1 Modeling marginal time series using ARMA-GARCH #########################

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


### 0.2 Modeling cross-sectional dependence  ##################################################

##' @title Modeling Dependence using Copulas and GMMNs
##' @param U matrix of pseudo-observations of the training data
##' @param GMMN.dim numeric vector of length at least two giving the dimensions
##'        of the input layer, the hidden layer(s) (if any) and the output layer;
##'        only needed if fitting GMMNs.
##' @param file character string with ending .rda
##' @return fitted copula or GMMN models
dependence_fit <- function(U, GMMN.dim, file)
{
    if (exists_rda(file, names = rm_ext(basename(file)))) {
        fitted.model <- read_rda(file = file, names = rm_ext(basename(file)))
        if(grepl("GMMN",file)) fitted.model <- to_callable(fitted.model) # unserialization for GMMNs
    } else { # fitting and saving copulas and GMMNs
        dep.types <- c("norm_ex", "norm_un", "t_ex", "t_un", "clayton", "gumbel", "GMMN") # types of models
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
                   "norm_un" = {
                       fitCopula(normalCopula(dim = d, dispstr = "un"),
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
                   "clayton" = {
                       fitCopula(claytonCopula(dim = d),
                                 data = U, method = "mpl", estimate.variance = FALSE)
                   },
                   "gumbel" = {
                       fitCopula(gumbelCopula(dim = d),
                                 data = U, method = "mpl", estimate.variance = FALSE)
                   },
                   "GMMN" = {
                       train_once(GMMN_model(GMMN.dim), data = U,
                                  batch.size = n, nepoch = nepo, file = file)
                   },
                   stop("Wrong 'method'"))
        ## Since train_once() already saves GMMN models we only need to save the copula models
        if(!grepl("GMMN", file)) save_rda(fitted.model, file = file)
    }
    fitted.model
}


### 0.3 Modeling multivariate time series  #####################################

##' @title Fitting All Dependence Models for Multivariate Time Series Data
##' @param X matrix containing time series data (i.e. training data)
##' @param series.strng character string specifying the financial time series to
##'        be used
##' @param train.period character string of type "YYYY-MM-DD_YYY_MM_DD"
##'        specifying the start and end date of the training period
##' @return list containing the fitted (list of) marginal models, matrix of
##'         pseudo-observations (training data for the dependence models)
##'         and list of all fitted dependence models (6 parametric copulas and
##'         one GMMN)
all_multivariate_ts_fits <- function(X, series.strng, train.period)
{
    ## File name for loading-saving ARMA-GARCH models associated with series.strng
    marginal.file <- paste0("ARMA_GARCH_", train.period,
                            "_", series.strng, ".rds")

    ## 1) First we feed training dataset to marginal_ts_fit()
    marginal.models <- marginal_ts_fit(X, file = marginal.file) # fit marginal time series models
    standard.resid <- lapply(marginal.models$fit, residuals, standardize = TRUE) # grab out standardized residuals
    Y <- as.matrix(do.call(merge, standard.resid)) # converting residuals to matrix data

    ## 2) No PCA here

    ## 3) Obtain pseudo-observations
    U <- pobs(Y)
    dimnames(U) <- NULL

    ## Setup
    dm <- dim(U)
    ntrn <- dm[1] # number of observations in the training dataset
    dim.in.out <- dm[2] # dimension of pseudo-observations used to train dependence models
    nbat <- ntrn # batch optimization

    ## 4) Fitting the copula models
    print("Fitting a Gumbel copula")
    model.cop.gumbel  <- dependence_fit(U,
                                        file = paste0("copula_","gumbel", "_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting a Clayton copula")
    model.cop.clayton <- dependence_fit(U,
                                        file = paste0("copula_","clayton","_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting an exchangeable normal copula")
    model.cop.norm.ex <- dependence_fit(U,
                                        file = paste0("copula_","norm_ex","_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting an unstructured normal copula")
    model.cop.norm.un <- dependence_fit(U,
                                        file = paste0("copula_","norm_un","_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting an exchangeable t copula")
    model.cop.t.ex    <- dependence_fit(U,
                                        file = paste0("copula_","t_ex","_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting an unstructured t copula")
    model.cop.t.un    <- dependence_fit(U,
                                        file = paste0("copula_","t_un","_dim_",dim.in.out,"_",series.strng,".rda"))

    ## 5) Fitting the GMMN model
    print("Training a GMMN")
    model.GMMN <- dependence_fit(U, GMMN.dim = c(dim.in.out, dim.hid, dim.in.out),
                                 file = paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",
                                               dim.in.out,"_ntrn_",ntrn,"_nbat_",nbat,
                                               "_nepo_",nepo,"_",series.strng,".rda"))

    ## 6) Results
    dependence.models = list(model.cop.gumbel  = model.cop.gumbel,
                             model.cop.clayton = model.cop.clayton,
                             model.cop.norm.ex = model.cop.norm.ex,
                             model.cop.norm.un = model.cop.norm.un,
                             model.cop.t.ex    = model.cop.t.ex,
                             model.cop.t.un    = model.cop.t.un,
                             model.GMMN    = model.GMMN)
    list(marginal = marginal.models, pobs.train = U, dependence = dependence.models) # return
}


### 0.4 Metric to compare the fitted dependence models (two-sample statistic) ##

##' @title B Realizations of Two-Sample GoF Statistics
##' @param pobs.train matrix containing the pseudo-observations (i.e. the training data)
##' @param dep.models list of all fitted dependence models (6 copulas, one GMMN)
##' @param series.strng character string specifying the financial time series to
##'        be used
##' @param B number of realization of the two-sample gof statistic to compute
##' @return (B, 7)-matrix containing the B replications of the Cramer-von Mises
##'         statistic for each of the 7 competing dependence models
gof2stats <- function(pobs.train, dep.models, series.strng, B = 100)
{
    dm <- dim(pobs.train)
    n <- dm[1] # number of observations in training data
    d <- dm[2] # dimension of training data

    ## File name for loading and saving realizations of gof 2 sample test statistics
    file <- paste0("gof2stat","_dim_",d,"_ngen_",ngen,"_B_",B,"_",series.strng,".rds")
    if(file.exists(file)) {
        gof.stats <- readRDS(file)
    } else {
        ## Compute B realizations of gof 2 sample test statistics
        set.seed(271) # for reproducibility
        gof.stats <- t(replicate(B, expr = {
            ## Generate samples from the fitted parametric copulas or GMMN
            U.models <- lapply(1:length(dep.models), function(i)
                if(grepl("cop", names(dep.models)[i])) {
                    rCopula(ngen, copula = dep.models[[i]]@copula)
                } else if(grepl("GMMN", names(dep.models)[i])) {
                    pobs(predict(dep.models[[i]][["model"]], x = matrix(rnorm(ngen * d), ncol = d)))
                })
            ## Compute the gof two-sample test statistics for each of the generated samples
            sapply(1:length(U.models), function(i) gofT2stat(U.models[[i]], pobs.train))
        }))
        colnames(gof.stats) <- names(dep.models) # preserve model names
        saveRDS(gof.stats, file = file)
    }
    gof.stats
}

##' @title Boxplots of the Two-Sample GoF Statistics
##' @param gof.stats return object of gof2stats()
##' @param ntrn training dataset sample size
##' @return invisible (boxplot by side-effect)
gof2stats_boxplot <- function(gof.stats, ntrn)
{
    ## Create a vector of names with each names corresponding to a fitted dependence model
    nms <- rep(NA, ncol(gof.stats))
    nms[which(grepl("gumbel",  colnames(gof.stats)))] <- "Gumbel"
    nms[which(grepl("clayton", colnames(gof.stats)))] <- "Clayton"
    nms[which(grepl("norm.ex", colnames(gof.stats)))] <- "Normal (ex)"
    nms[which(grepl("norm.un", colnames(gof.stats)))] <- "Normal (un)"
    nms[which(grepl("t.ex",    colnames(gof.stats)))] <- "t (ex)"
    nms[which(grepl("t.un",    colnames(gof.stats)))] <- "t (un)"
    nms[which(grepl("GMMN",    colnames(gof.stats)))] <- "GMMN"
    par(pty = 's')
    ## Boxplot
    boxplot(gof.stats, log = "y", names = nms,
            ylab = expression(S[list(n[trn],n[gen])]))
    mtext(substitute(B.~"replications, d ="~d.~", "~n[trn]~"="~ntrn.~", "~n[gen]~"="~ngen.,
                     list(B. = B, d. = d, ntrn. = ntrn, ngen. = ngen)),
          side = 4, line = 0.5, adj = 0)
}


### 0.5 Objective functions to study the variance reduction factor for #########

##' @title Compute B realizations of Each Objective Function Based on n GMMN PRS
##'        and GMMN QRS
##' @param gnn trained GMMN
##' @param marginal.fits list of fitted marginal models
##' @param B number of realizations
##' @param n sample size
##' @param randomize type or randomization used for QRS
##' @param S.t last available stock prices for financial objective functions
##' @param sig estimated marginal volatilities for financial objective functions
##' @param series.strng character string specifying the financial time series to
##'        be used
##' @return (<3 objective functions>, <2 random sampling types>, <B replications>)-array
##' @author Avinash Prasad
objective_functions <- function(gnn, marginal.fits, B, n, randomize, S.t, sig, series.strng)
{
    ## File name for loading and saving realizations of objective functions
    file <- paste0("objective_functions","_dim_",d,"_ngen_",n,"_B_",B,"_",series.strng,".rds")
    if (file.exists(file)) {
        readRDS(file)
    } else {
        ## Setup
        GMMNmod <- gnn[["model"]] # dependence model
        nus.fits <- sapply(marginal.fits, function(x) x@fit$coef[["shape"]]) # marginal fitted d.o.f.
        qInnovations <- function(u) sapply(1:d, function(j) # transformation back to fitted innovations
            sqrt((nus.fits[j]-2)/nus.fits[j]) * qt(u[,j], df = nus.fits[j]))
        qS <- function(u, S.t, t, T, r, sig) sapply(1:d, function(j) # transformation to margins from Black-Scholes model
            qlnorm(u[,j], meanlog = log(S.t[j]) + (r-sig[j]^2/2) * (T-t), sdlog = sqrt(sig[j]^2 * (T-t))))
        basket_call <- function(x, K, t, T, r) exp(-r*(T-t)) * mean(pmax(rowMeans(x) - K, 0)) # basked call objective function

        ## Fixed parameter choices for financial applications
        t <- 0 # now
        T <- 1 # maturity in years
        r <- 0.01 # risk-free annual interest rate

        ## Main function
        aux <- function(b) {
            ## Result object
            r. <- matrix(, nrow = 3, ncol = 2,
                        dimnames = list("Objective" = c("ES", "AC", "Basket call"),
                                        "RS" = c("GMMN PRS", "GMMN QRS")))

            ## Generate PRS and QRS
            set.seed(b) # for GMMN PRS
            U.PRS <- pobs(predict(GMMNmod, x = matrix(rnorm(n * d), ncol = d))) # GMMN PRS
            U.QRS <- pobs(predict(GMMNmod, x = qnorm(sobol(n, d = d, randomize = randomize, seed = b)))) # GMMN QRS

            ## Risk management applications: Use survival copula (as we model log-returns) and
            ## map to fitted t innovations
            Z.PRS <- qInnovations(1 - U.PRS)
            Z.QRS <- qInnovations(1 - U.QRS)

            ## 1) ES_alpha
            ## Note: Survival copula used since log-returns are modeled
            level <- 0.99
            r.[1,] <- c(ES_np(Z.PRS, level = level), ES_np(Z.QRS, level = level))

            ## 2) CA_alpha for first risk according to Euler principle
            r.[2,] <- c(alloc_np(Z.PRS, level = level)$allocation[1],
                       alloc_np(Z.QRS, level = level)$allocation[1])

            ## Financial applications: Use log-returns and map them to fitted log-normal
            ## margins as in Black-Scholes framework
            X.PRS <- qS(U.PRS, S.t = S.t, t = t, T = T, r = r, sig = sig)
            X.QRS <- qS(U.QRS, S.t = S.t, t = t, T = T, r = r, sig = sig)

            ## 3) Compute expected payoff of a basket call option with strike K
            r.[3,] <- c(basket_call(X.PRS, K = K.basket, t = t, T = T, r = r),
                       basket_call(X.QRS, K = K.basket, t = t, T = T, r = r))

            ## Return
            r.
        }

        ## Replications
        raw <- lapply(seq_len(B), function(b) aux(b))
        res <- simplify2array(raw) # convert list of 2-arrays to 3-array (dimension 'B' is added as last component)
        names(dimnames(res))[3] <- "Replication" # update name of dimnames
        dimnames(res)[[3]] <- 1:B # update dimnames

        ## Save results object
        saveRDS(res, file = file)
        res
    }
}

##' @title Boxplots of Objective Function realizations
##' @param obj.vals return object of objective_functions()
##' @param name Name of objective function
##' @return invisible (boxplot by side-effect)
VRF_boxplot <- function(obj.vals,name)
{
    ## Retrieve objective value realizations and compute variances
    varP <- var(GPRS <- obj.vals["GMMN PRS",])
    varQ <- var(GQRS <- obj.vals["GMMN QRS",])

    ## Compute the VRF and % improvements w.r.t. PRS
    VRF.Q <- formatC(varP / varQ,digits=2,format="f") # VRF for QRS
    PIM.Q <- formatC((varP - varQ) / varP * 100,digits=2,format="f") # % improvement for QRS
    
    ## Create labels for y-axis depending on the objective function
    ylabel <- if(grepl("ES",name)) {
      substitute(ES[alpha], list(alpha = 0.99))
      } else if (grepl("AC", name)) {
      substitute(AC[1,alpha], list(alpha = 0.99)) 
     } else if (grepl("Basket call", name)) {
      substitute("Basket call payoff, strike ="~K, list(K = K.basket))
    }
    
    
    
    ## Box plot
    par(pty = 's')
    boxplot(list(GPRS = GPRS, GQRS = GQRS),
            names=c("GMMN PRS","GMMN QRS"),ylab=ylabel)
    mtext(substitute(B.~"replications, d ="~d.~", "~n[gen]~"="~n.~", VRF (% improvements)"~VQ~"("~PQ~"%)",
                     list(B. = B, d. = d, n. = ngen., VQ = VRF.Q,PQ= PIM.Q)),
          side = 4, line = 0.5, adj = 0)
}

##' @title Results for data application
##' @param tickers list of character strings representing ticker symbols for S&P 500 stocks
##' @return invisible (plots by side-effect)
main <- function(tickers){
  ## Construct final risk factors to work with
  X <- X.[,tickers]
  d <- ncol(X)
  
  ## For financial examples
  S <- S.[, tickers]
  S.t <- t(tail(S, n = 1)) # last available values
  X.past.two <- X[paste0(c(2014-01-01, train.period[2]), collapse = "/"), ]
  sig <- apply(X.past.two, 2, sd) # estimate marginal volas based on past two years
  K.basket <- round(1.005 * mean(S.t)) # strike for basket call
  
  ## Fitting
  series.strng <- paste0(tickers, collapse = "_")
  train.period.strng <- paste0(train.period, collapse = "_")
  fits <- all_multivariate_ts_fits(X, series.strng = series.strng, # fitting
                                   train.period = train.period.strng)
  marginal.models <- fits$marginal ## fitted marginal models
  U.trn <- fits$pobs.train # pobs of the standardized residuals
  dep.models <- fits$dependence # fitted dependence models
  
  ## Visual assessment of the pobs after removing the marginal time series
  if(doPDF) pdf(file = (file <- paste0("fig_scatter_dim_",d,"_",series.strng,".pdf")))
  par(pty = "s")
  pairs2(U.trn, pch = ".")
  if(doPDF) dev.off.crop(file)
  
  ## Computing two-sample gof test statistics
  B <- 100
  gof.stats <- gof2stats(U.trn, dep.models = dep.models, series.strng = series.strng,B=B)
  
  ## Visual assessment of the two-sample gof test statistics
  file <- paste0("fig_boxplot_gof2stat","_dim_",d,"_ngen_",ngen,"_B_",B,"_",series.strng,".pdf")
  if(doPDF) pdf(file = (file <- file), height = 9.5, width = 9.5)
  gof2stats_boxplot(gof.stats, ntrn = ntrn)
  if(doPDF) dev.off.crop(file)
  
  ## Computing realizations of objective functions using GMMN PRS and GMMNQ QRS samples
  B. <- 200 # number of replications
  ngen. <- 1e5 # sample size
  res <- objective_functions(dep.models$model.GMMN, marginal.fits = marginal.models$fit,
                             B=B.,n=ngen.,randomize = "Owen",S.t=S.t,sig=sig,
                             series.strng = series.strng)
  
  ###  Visual assessment of variance reduction effects of GMMN QRS vs GMMN PRS
  ### for three objective functions
  
  file <- paste0("fig_boxplot_ES99","_dim_",d,"_ngen_",ngen.,"_B_",B.,"_",series.strng,".pdf")
  if(doPDF) pdf(file = (file <- file), height = 9, width = 9)
  VRF_boxplot(res[1,,], name = dimnames(res)[[1]][1])
  if(doPDF) dev.off.crop(file)
  
  file <- paste0("fig_boxplot_AC1","_dim_",d,"_ngen_",ngen.,"_B_",B.,"_",series.strng,".pdf")
  if(doPDF) pdf(file = (file <- file), height = 9, width = 9)
  VRF_boxplot(res[2,,], name = dimnames(res)[[1]][2])
  if(doPDF) dev.off.crop(file)
  
  file <- paste0("fig_boxplot_basketcall","_dim_",d,"_ngen_",ngen.,"_B_",B.,"_",series.strng,".pdf")
  if(doPDF) pdf(file = (file <- file),height = 9, width = 9)
  VRF_boxplot(res[3,,], name = dimnames(res)[[1]][3])
  if(doPDF) dev.off.crop(file)
  
}


### 1 Retrieve financial time series data using qrmtools package ###############

### 1.1 Data handling ##########################################################

## Loading S&P 500 constituent dataset and filter out those with few NA
data("SP500_const")
train.period <- c("1995-01-01", "2015-12-31") # training time period
raw <- SP500_const[paste0(train.period, collapse = "/"),] # data
keep <- apply(raw, 2, function(x.) mean(is.na(x.)) <= 0.01) # keep those with <= 1% NA
S. <- na.fill(raw[, keep], fill = "extend") # fill NAs
X. <- returns(S.) # compute log-returns
ntrn <- nrow(X.) # Training data size (constant for all selected portfolio of stocks)

## 1.2  Select constituents for each portfolio we work with ############################

## 1.2.1 Portfolio 1: d = 3 ########
tickers.P3 <- c("INTC", "IBM", # technology
                   "AIG") # financial

## 1.2.2  Portfolio 2: d = 5 #######
tickers.P5 <- c("INTC", "ORCL", "IBM", # technology
                 "COF", "AIG") # financial

## 1.2.3 Portfolio 3: d = 10 #######
tickers.P10 <- c("INTC", "ORCL", "IBM", # technology
             "COF", "JPM", "AIG", # financial
             "MMM", "BA", "GE", "CAT") # industrial


### 2 Computing all results ############################################

## 2.1 Plots for portfolio of stocks identified in 1.2.1
main(tickers.P3)

## 2.2 Plots for portfolio of stocks identified in 1.2.2
main(tickers.P5)

## 2.3 Plots for portfolio of stocks identified in 1.2.3
main(tickers.P10)


