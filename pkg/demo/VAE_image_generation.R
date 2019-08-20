## By Marius Hofert and Avinash Prasad

## Demonstrates application of variational autoencorders (VAEs) for generating
## images for the Frey faces dataset (1965 images of Brendan Frey, each 28 x 20
## pixels) and the Fashion MNIST dataset (60000 images of Zalando, each 28 x 28; see
## https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/).
## Training was done with options --gres=gpu:1
## --cpus-per-task=16 --mem=32G on a GPU server with NVIDIA Tesla P100 GPUs
## with 'module load gcc python/3.6 r/3.5.0' and 'source ~/soft/keras_r/bin/activate'
## executed before training.


### Setup ######################################################################

## Packages
library(keras) # R interface to Keras (high-level neural network API)
library(tensorflow) # R interface to TensorFlow (numerical computation with tensors)
library(gnn) # for the VAE model setup and training
library(RnavGraphImageData) # for the Frey faces dataset


### 1 Frey faces dataset #######################################################

## Load the data (from RnavGraphImageData)
data(frey)

## Adjust the dataset (so that each row represents a sample) and
## standardize it (so that pixel values are mapped to [0,1]; this helps
## training the VAE)
x.frey <- t(frey) / 255 # standardization to {0, 1/255, ..., 255/255}
str(x.frey) # (1965, 560)-matrix; 1965 faces, each 28 * 20 = 560 pixels
ncol.face <- 20 # number of columns for each face

## Plot some of the faces (target distribution for the VAE)
opar <- par(mar = rep(0.2, 4))
n <- 40 # number of faces (reused below)
ncol <- 5 # number of columns (reused below)
layout(matrix(1:n, ncol = ncol, byrow = TRUE))
for (i in 1:n)
    plot(as.raster(t(matrix(x.frey[i,], nrow = ncol.face)))) # as.raster => (28, 20)-matrix of colors
layout(1)
par(opar)

## Training parameters
dim.in.out <- ncol(x.frey) # dimension of the input and output layers
dim.hid <- 300L # dimension of (single) hidden layer
dim.lat <- 2L # dimension of the latent layer
ntrn <- nrow(x.frey) # training dataset size (number of pseudo-random numbers from the copula)
nbat <- 100L # batch size for training (number of samples per stochastic gradient step)
nepo <- 300L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
stopifnot(dim.in.out >= 1, dim.hid >= 1, dim.lat >= 1, ntrn >= 1,
          1 <= nbat, nbat <= ntrn, nepo >= 1)

## Train the VAE
NNname <- paste0("facesFrey_VAE_gen_dim_",dim.in.out,"_",dim.hid,"_",dim.lat,
                 "_ntrn_",ntrn,"_nbat_",nbat,"_nepo_",nepo,".rda")
VAE <- VAE_train_once(dim = c(dim.in.out, dim.hid, dim.lat), data = x.frey,
                      batch.size = nbat, nepo = nepo, file = NNname)

TODO: fix appendix() and others by replacing 'nbat' by 'batch.size'; compare with demo
TODO: from here; fix train_once() *everywhere* to GMMN_train_once() and package to work with VAEs
TODO: include comments from old vignette; follow similarly as in GMMN vignette

## Train the VAE
if(exists_rda(objname, objnames = objname, package = "gnn")) { # get trained VAE generator
    VAE.frey$generator <- unserialize_model(read_rda(objname, file = objname, package = "gnn"),compile=FALSE)
} else { # train the VAE and save the corresponding generator
    VAE.frey$model %>% fit(x.frey, x.frey, epochs = nepo, batch_size = nbat)
    save_rda(serialize_model(VAE.frey$generator),
             file = paste0(objname,".rda"), names = objname)
}


##     VAE.frey$generator <- unserialize_model(read_rda(objname, file = objname, package = "gnn"),
##                                             compile=FALSE)
##     save_rda(serialize_model(VAE.frey$generator),
##              file = paste0(objname,".rda"), names = objname)

## Generate Frey-like faces from the fitted decoder (generator)
set.seed(271) # for reproducibility
N01.prior <- matrix(rnorm(n * dim.lat), ncol = dim.lat) # sample from the prior
x.frey. <- predict(VAE.frey$generator, x = N01.prior) # Frey sample

## Plot
opar <- par(mar = rep(0.2, 4))
layout(matrix(1:n, ncol = ncol, byrow = TRUE))
for (i in 1:n)
    plot(as.raster(t(matrix(x.frey.[i,], nrow = ncol.face)))) # as.raster => (28, 20)-matrix of colors
layout(1)
par(opar)


### 2 Fashion MNIST dataset ####################################################

TODO: adapt to the above once finished

fmnist <- dataset_fashion_mnist() # load the full fashion MNIST dataset from 'keras'
x.fmnist <- fmnist$train$x / 255 # standardize the dataset
x.fmnist <- t(apply(x.fmnist, 1, as.numeric)) # adjustment to create the appropriate traning dataset
ntrn <- nrow(x.fmnist) # training dataset size
dim.in <- ncol(x.fmnist) # dimension of training dataset
dim.out <- dim.in

## Plot
n.fmnist <- 10 # number of fashion-MNIST images to display
nrow.fmnist <- 28 # number of rows of a single fashion-MNIST image
layout(matrix(1:n.fmnist, ncol = 5, byrow = TRUE))
for (i in 1:n.fmnist)
    plot(as.raster(matrix(x.fmnist[i,], nrow = nrow.fmnist)))
layout(1)


## Training setup
nepo <- 100 # epochs (= passes through training dataset while updating NN parameters)
nbat <- 100 # training batch size (number of samples per stochastic gradient step)
dim.hidden <- 300 # dimension of the three hidden layers
dim.lat <- 2 # dimension of latent layer
VAE.fmnist <- VAE_model(c(dim.in, rep(dim.hidden, 3), dim.lat)) # setup the VAE
objname <- paste0("VAE_gen_dim_",dim.lat,"_3_",dim.hidden,"_",dim.out,"_ntrn_",ntrn,
                  "_nbat_",nbat,"_nepo_",nepo,"_fmnist")
## Train the VAE
if(exists_rda(objname, objnames = objname, package = "gnn")) { # get trained VAE generator
    VAE.fmnist$generator <- unserialize_model(read_rda(objname, file = objname, package = "gnn"),compile=FALSE)
} else { # train the VAE and save the corresponding generator
    VAE.fmnist$model %>% fit(x.fmnist, x.fmnist, epochs = nepo, batch_size = nbat)
    save_rda(serialize_model(VAE.fmnist$generator),
             file = paste0(objname,".rda"), names = objname)
}

## Sample from the latent distribution
N01.latent <- matrix(rnorm(n.fmnist * dim.lat), ncol = dim.lat)
# Generate Frey faces from the fitted VAE
x.fmnist.VAE <- predict(VAE.fmnist$generator, N01.latent)
## Plot
layout(matrix(1:n.fmnist, ncol = 5, byrow = TRUE))
for (i in 1:n.fmnist)
    plot(as.raster(matrix(x.fmnist.VAE[i,], nrow = nrow.fmnist)))
layout(1)
