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

## Load the data (from 'RnavGraphImageData')
data(frey)

## Adjust the dataset (so that each row represents a sample) and
## standardize it (so that pixel values are mapped to [0,1]; this helps
## training the VAE)
x.frey <- t(frey) / 255 # standardization to {0, 1/255, ..., 255/255}
str(x.frey) # (1965, 560)-matrix; 1965 faces, each 28 * 20 = 560 pixels
nc.img <- 20 # number of columns of each picture

## Plot some of the images (target distribution for the VAE)
opar <- par(mar = rep(0.2, 4))
n <- 40 # number of images (reused below)
nc <- 5 # number of columns (reused below)
layout(matrix(1:n, ncol = nc, byrow = TRUE))
for (i in 1:n)
    plot(as.raster(t(matrix(x.frey[i,], nrow = nc.img)))) # as.raster => (28, 20)-matrix of colors
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
NNname <- paste0("VAE_imggen_Frey_dim_",dim.in.out,"_",dim.hid,"_",dim.lat,
                 "_ntrn_",ntrn,"_nbat_",nbat,"_nepo_",nepo,".rda")
GNN <- VAE_model(c(dim.in.out, dim.hid, dim.lat)) # model setup
system.time(VAE <- train_once(GNN, data = x.frey, # train once
                              batch.size = nbat, nepo = nepo, file = NNname)) # training ~= 20s

## Generate Frey-like images from the fitted generator
set.seed(271) # for reproducibility
N01.prior <- matrix(rnorm(n * dim.lat), ncol = dim.lat) # sample from the prior
x.frey. <- predict(VAE$generator, x = N01.prior) # # sample via the generator

## Plot
opar <- par(mar = rep(0.2, 4))
layout(matrix(1:n, ncol = nc, byrow = TRUE))
for (i in 1:n)
    plot(as.raster(t(matrix(x.frey.[i,], nrow = nc.img)))) # as.raster => (28, 20)-matrix
layout(1)
par(opar)


### 2 Fashion MNIST dataset ####################################################

## Load the data (from online database behind 'keras')
fmnist <- dataset_fashion_mnist()
str(fmnist) # (60000, 28, 28)-array
x.fmnist <- t(apply(fmnist$train$x, 1, function(img) as.vector(t(img)) / 255)) # flattening and standardization
str(x.fmnist) # (60000, 784)-matrix; 60000 fashion pieces, each 28 * 28 = 728 pixels
nc.img <- 28 # number of columns of each picture

## Plot some of the images (target distribution for the VAE)
opar <- par(mar = rep(0.2, 4))
layout(matrix(1:n, ncol = nc, byrow = TRUE))
for (i in 1:n)
    plot(as.raster(t(matrix(x.fmnist[i,], nrow = nc.img)))) # as.raster => (28, 28)-matrix of colors
layout(1)
par(opar)

## Training parameters
dim.in.out <- ncol(x.fmnist) # dimension of the input and output layers
dim.hid <- 300L # dimension of (single) hidden layer
dim.lat <- 2L # dimension of the latent layer
ntrn <- nrow(x.fmnist) # training dataset size (number of pseudo-random numbers from the copula)
nbat <- 100L # batch size for training (number of samples per stochastic gradient step)
nepo <- 100L # number of epochs (one epoch = one pass through the complete training dataset while updating the GNN's parameters)
stopifnot(dim.in.out >= 1, dim.hid >= 1, dim.lat >= 1, ntrn >= 1,
          1 <= nbat, nbat <= ntrn, nepo >= 1)

## Train the VAE
NNname <- paste0("VAE_imggen_fMNIST_dim_",dim.in.out,"_",
                 paste(rep(dim.hid, 3), collapse = "_"),"_",
                 dim.lat,"_ntrn_",ntrn,"_nbat_",nbat,"_nepo_",nepo,".rda")
GNN <- VAE_model(c(dim.in.out, rep(dim.hid, 3), dim.lat)) # model setup
system.time(VAE <- train_once(GNN, data = x.fmnist, # train once
                              batch.size = nbat, nepo = nepo, file = NNname)) # training ~= 8.5min

## Generate fashion-MNIST-like images from the fitted generator
set.seed(271) # for reproducibility
N01.prior <- matrix(rnorm(n * dim.lat), ncol = dim.lat) # sample from the prior
x.fmnist. <- predict(VAE$generator, x = N01.prior) # sample via the generator

## Plot
opar <- par(mar = rep(0.2, 4))
layout(matrix(1:n, ncol = nc, byrow = TRUE))
for (i in 1:n)
    plot(as.raster(t(matrix(x.fmnist.[i,], nrow = nc.img)))) # as.raster => (28, 28)-matrix
layout(1)
par(opar)
