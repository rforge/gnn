### Generative moment matching network #########################################

##' @title Generative Moment Matching Network (GMMN)
##' @param dim numeric vector of length at least two giving the dimensions of
##'        the input layer, the hidden layer(s) (if any) and the output layer
##' @param activation character vector of length length(dim) - 1 specifying the
##'        activation functions for all hidden layers and the output layer
##'        (in this order); note that the input layer does not have an
##'        activation function.
##' @param batch.norm logical indicating whether batch normalization
##'        layers are to be added after each hidden layer.
##' @param dropout.rate numeric value in [0,1] specifying the fraction of input
##'        to be dropped; see the rate parameter of layer_dropout().
##'        Only if positive, dropout layers are added after each hidden layer.
##' @param nGPU non-negative integer specifying the number of GPUs available
##'        if the GPU version of TensorFlow is installed. If positive, a
##'        (special) multiple GPU model for data parallelism is instantiated.
##'        Note that for multi-layer perceptrons on a few GPUs, this model does
##'        not yet yield any scale-up computational factor (in fact, currently
##'        very slightly negative scale-ups are likely due to overhead costs).
##' @param ... additional arguments passed to the underlying loss function;
##'        at the moment, this can be the bandwith parameter 'bandwidth'
##'        which is passed on by loss() to Gaussian_mixture_kernel().
##' @return List with Keras model of the GMMN and additional information
##' @note - Could at some point have an argument 'kernel.type' which specifies
##'         a different kernel.
##'       - The respective parameters are then passed via '...' (as we do for
##'         loss() at the moment).
##'       - Make sure that the resulting NN is always a GMMN.
GMMN_model <- function(dim, activation = c(rep("relu", length(dim) - 2), "sigmoid"),
                       batch.norm = FALSE, dropout.rate = 0, nGPU = 0, ...)
{
    ## Basic input checks and definitions
    num.lay <- length(dim) # number of layers (including input and output layer)
    len.activ <- length(activation) # has to be of length num.lay - 1 (input layer has no activation function)
    stopifnot(num.lay >= 2, is.numeric(dim), dim >= 1,
              len.activ == num.lay - 1, is.character(activation),
              is.logical(batch.norm), 0 <= dropout.rate, dropout.rate <= 1,
              nGPU >= 0)
    num.hidden <- num.lay - 2 # number of hidden layers (= number of layers - input - output); can be 0
    ind.hid.lay <- seq_len(num.hidden) # note: num.hidden can be 0

    ## 1) Set up layers
    ## 1.1) Input layer
    in.lay <- layer_input(shape = dim[1])

    ## 1.2) Hidden layers (multi-layer perceptron)
    for(i in ind.hid.lay) {
        hid.lay <- layer_dense(if(i == 1) in.lay else hid.lay,
                               units = dim[1 + i], # dimensions of hidden layers (input layer is in 1st component)
                               activation = activation[i]) # 'activation' starts with hidden layers
        if(batch.norm) hid.lay <- layer_batch_normalization(hid.lay)
        if(dropout.rate > 0) hid.lay <- layer_dropout(hid.lay, rate = dropout.rate)
    }

    ## 1.3) Output layer
    out.lay <- layer_dense(hid.lay, units = dim[1 + num.hidden + 1],
                           activation = activation[num.hidden + 1])

    ## 2) Define the GMMN
    model <- if(nGPU > 0) {
                 ## To ensure memory is hosted on the CPU and not on the GPU
                 ## see https://keras.rstudio.com/reference/multi_gpu_model.html
                 with(tf$device("/cpu:0"), {
                     model. <- keras_model(in.lay, out.lay)
                 })
                 multi_gpu_model(model., gpus = nGPU) # replicated model on different GPUs
             } else keras_model(in.lay, out.lay)

    ## 3) Loss function
    ##    Note: - Required to be provided like that as otherwise:
    ##            "Error in loss(x, y = out.lay, ...) : object 'x' not found"
    ##          - unserialize_model() calls need to provide 'custom_objects = c(loss = loss)'
    loss_fn <- function(x, y = out.lay)
        loss(x, y = y, type = "MMD", ...) # GMMNs need to have "MMD" (otherwise not GMMNs)

    ## 4) Compile the model
    model %>% compile(optimizer = "adam", loss = loss_fn)

    ## Return
    list(model = model, type = "GMMN", dim = dim, activation = activation,
         batch.norm = batch.norm, dropout.rate = dropout.rate,
         dim.train = NA, batch.size = NA, nepoch = NA)
}
