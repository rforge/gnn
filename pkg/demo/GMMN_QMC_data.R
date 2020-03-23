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
package <- NULL # TODO: needed?
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
    if (exists_rda(file, names = rm_ext(basename(file)), package = package)) {
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
                                  batch.size = n, nepoch = nepo, file = file,
                                  package = package)
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
##' @param train.period character vector of length 2 with entries "YYYY-MM-DD"
##'        specifying the start and end date of the training period
##' @return list containing the fitted (list of) marginal models, matrix of
##'         pseudo-observations (training data for the dependence models)
##'         and list of all fitted dependence models (6 parametric copulas and
##'         one GMMN)
all_multivariate_ts_fits <- function(X, series.strng, train.period)
{
    ## File name for loading-saving ARMA-GARCH models associated with series.strng
    marginal.file <- paste0("ARMA_GARCH_", paste0(train.period, collapse = "_"),
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
    model.gumbel  <- dependence_fit(U,
                                    file = paste0("copula_","gumbel", "_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting a Clayton copula")
    model.clayton <- dependence_fit(U,
                                    file = paste0("copula_","clayton","_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting an exchangeable normal copula")
    model.norm.ex <- dependence_fit(U,
                                    file = paste0("copula_","norm_ex","_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting an unstructured normal copula")
    model.norm.un <- dependence_fit(U,
                                    file = paste0("copula_","norm_un","_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting an exchangeable t copula")
    model.t.ex    <- dependence_fit(U,
                                    file = paste0("copula_","t_ex","_dim_",dim.in.out,"_",series.strng,".rda"))
    print("Fitting an unstructured t copula")
    model.t.un    <- dependence_fit(U,
                                    file = paste0("copula_","t_un","_dim_",dim.in.out,"_",series.strng,".rda"))

    ## 5) Fitting the GMMN model
    print("Training a GMMN")
    model.GMMN <- dependence_fit(U, GMMN.dim = c(dim.in.out, dim.hid, dim.in.out),
                                 file = paste0("GMMN_dim_",dim.in.out,"_",dim.hid,"_",
                                               dim.in.out,"_ntrn_",ntrn,"_nbat_",nbat,
                                               "_nepo_",nepo,"_",series.strng,".rda"))

    ## 6) Results
    dependence.models = list(model.gumbel  = model.gumbel,
                             model.clayton = model.clayton,
                             model.norm.ex = model.norm.ex,
                             model.norm.un = model.norm.un,
                             model.t.ex    = model.t.ex,
                             model.t.un    = model.t.un,
                             model.GMMN    = model.GMMN)
    list(marginal = marginal.models, pobs.train = U, dependence = dpendence.models) # return
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
                    pobs(rCopula(ngen, copula = dep.models[[i]]@copula))
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

    ## Boxplot
    boxplot(gof.stats, log = "y", names = nms,
            ylab = expression(S[list(n[gen],n[trn])]))
    mtext(substitute(B.~"replications, d ="~d.~", "~n[gen]~"="~ngen.~", "~n[trn]~"="~ntrn.,
                     list(B. = B, d. = d, ngen. = ngen, ntrn. = ntrn)),
          side = 4, line = 0.5, adj = 0)
}


### 1 Retrieve financial time series data using qrmtools package ###############

### 1.1 Data handling ##########################################################

## Loading S&P 500 constituent dataset and filter out those with few NA
data("SP500_const")
train.period <- c("1995-01-01", "2015-12-31") # training time period
raw <- SP500_const[paste0(train.period, collapse = "/"),] # data
keep <- apply(raw, 2, function(x.) mean(is.na(x.)) <= 0.01) # keep those with <= 1% NA
S. <- raw[, keep] # data we keep
S <- na.fill(S., fill = "extend") # fill NAs
X. <- returns(S) # compute negative log-returns

## Select constituents we work with
tickers <- c("INTC", "ORCL", "IBM", # technology
             "COF","JPM","AIG", # financial
             "MMM","BA","GE","CAT") # industrial
## tickers <- c("AAPL","MSFT","IBM", # technology
##              "BAC","C") # financial
X <- X.[, tickers] # final risk factor changes we work with
ntrn <- nrow(X)
d <- ncol(X)


### 1.2 Fitting and comparison of multivariate time series models ##############

## Fitting
series.strng <- paste0("SP500_",paste0(tickers, collapse = "_"))
train.period.strng <- paste0(train.period, collapse = "_")
fits <- all_multivariate_ts_fits(X, series.strng = series.strng, # fitting
                                 train.period = train.period.strng)
U.trn <- fits$pobs.train # pobs of the standardized residuals
dep.models <- fits$dependence # fitted dependence models

## Visual assessment of the pobs after removing the marginal time series
if(doPDF) pdf(file = (file <- paste0("fig_scatter_dim_",d,"_",series.strng,".pdf")))
pairs2(U.trn, pch = ".")
if(doPDF) dev.off.crop(file)

## Computing two-sample gof test statistics
gof.stats <- gof2stats(U.trn, dep.models = dep.models, series.strng = series.strng)

## Visual assessment of the two-sample gof test statistics
file <- paste0("fig_boxplot_gof2stat","_dim_",d,"_ngen_",ngen,"_B_",B,"_",series.strng,".pdf")
if(doPDF) pdf(file = (file <- file), height = 9.5, width = 9.5)
gof2stats_boxplot(gof.stats, ntrn = ntrn)
if(doPDF) dev.off.crop(file)
