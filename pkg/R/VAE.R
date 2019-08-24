### Auxiliaries ################################################################

##' @title Sampling from (Multivariate) Normal Distributions for Latent Layers
##' @param x 2d tensor with shape (batch_size, 2 * dim) representing a
##'        concatenated Keras layer: (mean, log.var)
##' @param sd standard deviation for the normal distribution (prior)
##' @param dim dimension of latent layer
##' @return 2d tensor with shape (batch_size, dim) of samples
##'         from N(mean, sd^2 * var) where var = exp(log.var).
##' @author Marius Hofert and Avinash Prasad
tf_rnorm <- function(x, sd, dim)
{
    ii <- seq_len(dim)
    mu    <- x[, ii]
    lsig2 <- x[, dim + ii] # sig2 = exp(lsig2)
    mu + k_exp(lsig2) * k_random_normal(shape = c(k_shape(mu)[[1]]), # shape of tensor (that of mu)
                                        mean = 0, stddev = sd) # mu (as tensor) + sigma (as tensor) * N(0,sd)
}

##' @title Kullback-Leibler Divergence Loss Function
##' @param mean 2d tensor with shape (batch_size, dimension of latent layer)
##'        containing the means
##' @param log.var 2d tensor with (batch_size, dimension of latent layer)
##'        containing logarithmic variances
##' @return 0d tensor containing the K-L loss value
##' @author Marius Hofert and Avinash Prasad
##' @note The function computes the K-L divergence between two normal distributions
##'       N(mean, exp(log.var)) and N(0, I), where I is the identity matrix of
##'       dimension equal to the dimension of the latent layer.
##'       Note:
##'       'mean' and 'log.var' of the former normal distribution is not constant,
##'       it appears to change with each sample. So the dimension of N(mean, exp(log.var))
##'       is the one of the latent layer, but we minimize a different normal
##'       distribution with each 'observation'. The trick is to think about the
##'       loss function being applied observation-by-observation. So, in VAE_model(),
##'       we minimize distance between ntrn different normal distributions and
##'       the standard normal.
KL <- function(mean, log.var)
    -0.5 * k_mean(1 + log.var - k_square(mean) - k_exp(log.var), axis = -1L)


### Variational autoencoder ####################################################

##' @title Variational Autoencoder (VAE)
##' @param dim numeric vector of length at least two giving the dimensions of
##'        the input (= output) layers (equal), the hidden layer(s) (if any) and
##'        the latent layer.
##' @param activation character vector of length length(dim) - 1 specifying the
##'        activation functions for all hidden layers and the output layer
##'        (in this order); note that the input layer does not have an
##'        activation function.
##' @param batch.norm logical indicating whether batch normalization
##'        layers are to be added after each hidden layer.
##' @param dropout.rate numeric value in [0,1] specifying the fraction of input
##'        to be dropped; see the rate parameter of layer_dropout().
##'        Only if positive, dropout layers are added after each hidden layer.
##' @param sd standard deviation of the normal distribution (prior).
##' @param loss.type character string indicating the type of reconstruction loss;
##'        see ?loss.
##' @param nGPU non-negative integer specifying the number of GPUs available
##'        if the GPU version of TensorFlow is installed. If positive, a
##'        (special) multiple GPU model for data parallelism is instantiated.
##'        Note that for multi-layer perceptrons on a few GPUs, this model does
##'        not yet yield any scale-up computational factor (in fact, currently
##'        very slightly negative scale-ups are likely due to overhead costs).
##' @param ... additional arguments passed to the underlying loss function;
##'        at the moment, this can be the bandwith parameter 'bandwidth'
##'        which is passed on by loss() to Gaussian_mixture_kernel().
##' @return List with Keras model of the VAE and additional information,
##'         including the model's encoder and generator
##' @author Marius Hofert and Avinash Prasad
##' @note - For a picture, see https://www.doc.ic.ac.uk/~js4416/163/website/autoencoders/variational.html
##'       - There is always one latent layer and typically (probably always) at
##'         least one hidden layer.
##'       - For an example of a simple VAE (based on Keras), see
##'         https://keras.rstudio.com/articles/examples/variational_autoencoder.html
VAE_model <- function(dim, activation = c(rep("relu", length(dim) - 2), "sigmoid"),
                      batch.norm = FALSE, dropout.rate = 0, sd = 1,
                      loss.type = c("MSE", "binary.cross", "MMD"), nGPU = 0, ...)
{
    ## Basic input checks and definitions
    num.lay <- length(dim) # number of layers (including input (= output) layer)
    len.activ <- length(activation) # has to be of length num.lay - 1 (input layer has no activation function)
    stopifnot(num.lay >= 2, is.numeric(dim), dim >= 1,
              len.activ == num.lay - 1, is.character(activation),
              is.logical(batch.norm), 0 <= dropout.rate, dropout.rate <= 1,
              sd > 0, nGPU >= 0)
    loss.type <- match.arg(loss.type)
    num.hidden <- num.lay - 2 # number of hidden layers (= number of layers - input (= output) - latent); can be 0
    ind.hid.lay <- seq_len(num.hidden) # note: num.hidden can be 0

    ## 1) Set up layers
    ## 1.1) Input layer
    in.lay <- layer_input(shape = dim[1])

    ## 1.2) Encoder layers (multi-layer perceptron)
    ##      Note: As in GMMN_model(), just called 'enc.lay' here instead of 'hid.lay'
    for (i in ind.hid.lay) {
        enc.lay <- layer_dense(if(i == 1) in.lay else enc.lay,
                               units = dim[1 + i], # dimensions of hidden layers (input (= output) layer is in 1st component)
                               activation = activation[i]) # 'activation' starts with hidden layers
        if(batch.norm) enc.lay <- layer_batch_normalization(enc.lay)
        if(dropout.rate > 0)    enc.lay <- layer_dropout(enc.lay, rate = dropout.rate)
    }

    ## 1.3) Latent layers
    ##      Note: - The encoder essentially provides a mapping from the original data (in.lay)
    ##              to two vectors (mean and log.var) which are layers here
    ##            - We map the encoder network to log.var instead of var as a workaround to
    ##              avoid the need of a positivity constraint on the output of the encoder
    ##              neural network. While the default 'relu' activation function already enforces
    ##              this constraint, it may still be helpful to map to log.var for numerical
    ##              stability of the optimization procedures.
    mean    <- layer_dense(enc.lay, units = dim[1 + num.hidden + 1]) # mean vector
    log.var <- layer_dense(enc.lay, units = dim[1 + num.hidden + 1]) # log-variance vector

    ## Next, we sample from N(mu, Sigma) (dimension equal to that of the latent layers)
    ## with mu = mean and Sigma = sd^2 * var, where var denotes a variance vector
    ## (hence Sigma is constrained to be a diagonal covariance matrix).
    ## Note: - Since stochastic gradient descent can handle stochastic inputs but not stochastic
    ##         units within the neural network, we use the "reparameterization trick" (see
    ##         https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
    ##         (based on the location-scale property of normal distributions) to move the
    ##         sampling outside of the variational autoencoder network architecture.
    ##       - Because of using layer_lambda(), we can't save VAEs via serialize_model();
    ##         same error appears as on https://github.com/rstudio/keras/issues/412
    lat.lay <- layer_concatenate(list(mean, log.var)) %>%
        layer_lambda(f = function(x) tf_rnorm(x, sd, dim = dim[1+num.hidden+1])) # sampling from N(mean, sd^2*var))

    ## 1.4) Define the encoder
    ## Why only for 'mean' and not for 'log.var'?
    ## This seems to be the common output of the encoder. There seems to be no good explanation.
    ## Note that this means each 'image' (observation) has a different 'mean' associated with it.
    encoder <- keras_model(in.lay, mean)

    ## 1.5) Decoder layers
    ##      Note: Store decoder layers to re-use for generator (see below); the decoder is
    ##            a mirror image of the encoder, but batch normalization and dropout layers
    ##            must only be present in the encoder to ensure proper training.
    for (i in ind.hid.lay) {
        ## We first define the (hidden) layers of the decoder network (denoted by
        ## decoder_1,..., decoder_{num.hidden}) with the appropriate dimensions
        ## and activation functions.
        assign(paste0("decoder", i, sep = "_"),
               layer_dense(units = dim[2 + num.hidden - i],
                           activation = activation[1+num.hidden - i]))
        ## We now instantiate these (hidden) layers by establishing the
        ## connection to the latent layer.
        lay.dec <- get(paste0("decoder", i, sep = '_'))(if(i == 1) lat.lay else lay.dec)
        ## Hence we have defined the 'decoded' (hidden) layers.
    }

    ## 1.6) Output layer
    decoder_out <- layer_dense(units = dim[1], activation = activation[num.hidden + 1])
    ## We now instantiate the output of the decoder network by establishing the
    ## connection to the (last) hidden layer
    out.lay <- decoder_out(lay.dec)

    ## 2) Define the VAE
    model <- if(nGPU > 0) {
                 ## To ensure memory is hosted on the CPU and not on the GPU
                 ## see https://keras.rstudio.com/reference/multi_gpu_model.html
                 with(tf$device("/cpu:0"), {
                     model. <- keras_model(in.lay, out.lay)
                 })
                 multi_gpu_model(model., gpus = nGPU) # replicated model on different GPUs
             } else keras_model(in.lay, out.lay)

    ## 3) Loss function
    ## Note: - Required to be provided like that; see GMMN_model()
    ##       - unserialize_model() calls need to provide 'custom_objects = c(loss = RKL)'
    loss_fn <- function(x, y = out.lay)
        KL(mean = mean, log.var = log.var) +
            (dim[1]/1.0) * loss(x, y = y, type = loss.type, ...) # multiply reconstruction loss by dimension (of output dataset) to ensure the two losses are of the same order

    ## 4) Compile the model
    model %>% compile(optimizer = "adam", loss = loss_fn)

    ## 5) Generator
    ##    Note: The generator takes input samples from the normal (prior) distribution
    ##          and passes it through the decoder network

    ## Input layer for the generator (same dimension as the latent layer)
    gen.in.lay <- layer_input(shape = dim[1 + num.hidden + 1])
    ## Use the previously defined (hidden) layers of the decoder network to instantiate
    ## the generator network and establish a connection to the generator input layer
    for (i in ind.hid.lay)
        gen.dec <- get(paste0("decoder", i, sep='_'))(if(i == 1) gen.in.lay else gen.dec)
    ## Use the previously defined output layer of the decoder network to instantiate
    ## the generator output and establish a connection to the generator (hidden) layers
    gen.out.lay <- decoder_out(gen.dec) # instantiate output layer
    generator <- keras_model(gen.in.lay, gen.out.lay) # define generator

    ## Return
    list(model = model, encoder = encoder, generator = generator, type = "VAE",
         dim = dim, activation = activation, batch.norm = batch.norm,
         dropout.rate = dropout.rate, sd = sd, loss.type = loss.type,
         dim.train = NA, batch.size = NA, nepoch = NA)
}
