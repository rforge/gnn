### FNN basic functions ########################################################

##' @title Determine the Number of Parameters
##' @param x object of class "keras.engine.training.Model" or "gnn_GNN"
##' @return 3-vector with the number of trainable, non-trainable and the total
##'         number of parameters
##' @author Marius Hofert
##' @note See https://stackoverflow.com/questions/47312219/what-is-the-definition-of-a-non-trainable-parameter
##'       for the meaning of 'trainable' vs 'non-trainable'
nparam_FNN <- function(x)
{
    if(inherits(x, "gnn_GNN")) x <- x[["model"]]
    num.tot.params <- count_params(x) # total number of parameters
    ## For the trainable parameters, there's no function, so we extract the information
    x.as.char <- as.character(x)
    char.vec <- strsplit(x.as.char, split = "\n")[[1]]
    pos <- which(sapply(char.vec, function(x) grepl("Trainable params:", x)))
    rm.str <- sub("Trainable params: ", "", char.vec[pos])
    num.trainable.params <- as.integer(gsub(",", "", rm.str))
    ## Result
    c("trainable" = num.trainable.params,
      "non-trainable" = num.tot.params - num.trainable.params,
      "total" = num.tot.params)
}


### FNN constructor ############################################################

##' @title Feedforward Neural Network (FNN) Constructor
##' @param dim integer vector of length at least two giving the dimensions
##'        of the input layer, the hidden layer(s) (if any) and the output layer
##' @param activation character vector of length length(dim) - 1 specifying
##'        the activation functions for all hidden layers and the output layer
##'        (in this order); note that the input layer does not have an
##'        activation function.
##' @param "sigmoid")
##' @param batch.norm logical indicating whether batch normalization
##'        layers are to be added after each hidden layer.
##' @param dropout.rate numeric value in [0,1] specifying the fraction of input
##'        to be dropped; see the rate parameter of layer_dropout().
##'        Only if positive, dropout layers are added after each hidden layer.
##' @param loss.fun loss function specified as character string or function.
##' @param n.GPU non-negative integer specifying the number of GPUs available
##'        if the GPU version of TensorFlow is installed. If positive, a
##'        (special) multiple GPU model for data parallelism is instantiated.
##'        Note that for multi-layer perceptrons on a few GPUs, this model does
##'        not yet yield any scale-up computational factor (in fact, currently
##'        very slightly negative scale-ups are likely due to overhead costs).
##' @param ... additional arguments passed to the underlying loss function;
##'        at the moment, this can be the bandwith parameter 'bandwidth'
##'        which is passed on by loss() to Gaussian_mixture_kernel().
##' @return An object of class "gnn_FNN" (inheriting from "gnn_GNN" and
##'         "gnn_Model") being a list with the Keras model of the FNN
##'         and additional information
##' @note - Could at some point have an argument 'kernel.type' which specifies
##'         a different kernel.
##'       - The respective parameters are then passed via '...' (as we do for
##'         loss() at the moment).
##'       - names(<FNN>$model) provides slots of "keras.engine.training.Model" object
FNN <- function(dim = c(2, 2), activation = c(rep("relu", length(dim) - 2), "sigmoid"),
                batch.norm = FALSE, dropout.rate = 0, loss.fun = "MMD", n.GPU = 0, ...)
{
    ## Basic input checks and definitions
    num.lay <- length(dim) # number of layers (including input and output layer)
    len.activ <- length(activation) # has to be of length num.lay - 1 (input layer has no activation function)
    stopifnot(num.lay >= 2, is.numeric(dim), dim >= 1,
              len.activ == num.lay - 1, is.character(activation),
              is.logical(batch.norm), 0 <= dropout.rate, dropout.rate <= 1,
              n.GPU >= 0)
    storage.mode(dim) <- "integer" # see ?as.integer
    num.hidden <- num.lay - 2 # number of hidden layers (= number of layers - input - output); can be 0

    ## 1) Set up layers
    ## 1.1) Input layer
    in.lay <- layer_input(shape = dim[1])

    ## 1.2) Hidden layers (multi-layer perceptron)
    for(i in seq_len(num.hidden)) {
        hid.lay <- layer_dense(if(i == 1) in.lay else hid.lay,
                               units = dim[1 + i], # dimensions of hidden layers (input layer is in 1st component)
                               activation = activation[i]) # 'activation' starts with hidden layers
        if(batch.norm) hid.lay <- layer_batch_normalization(hid.lay)
        if(dropout.rate > 0) hid.lay <- layer_dropout(hid.lay, rate = dropout.rate)
    }

    ## 1.3) Output layer
    out.lay <- layer_dense(if(num.hidden == 0) in.lay else hid.lay,
                           units = dim[1 + num.hidden + 1],
                           activation = activation[num.hidden + 1])

    ## 2) Define the FNN
    model <- if(n.GPU > 0) {
                 ## To ensure memory is hosted on the CPU and not on the GPU
                 ## see https://keras.rstudio.com/reference/multi_gpu_model.html
                 with(tf$device("/cpu:0"), {
                     model. <- keras_model(in.lay, out.lay)
                 })
                 multi_gpu_model(model., gpus = n.GPU) # replicated model on different GPUs
             } else keras_model(in.lay, out.lay)

    ## 3) Loss function
    ##    Note: - Required to be provided like that as otherwise:
    ##            "Error in loss(x, y = out.lay, ...) : object 'x' not found"
    ##          - unserialize_model() calls need to provide 'custom_objects = c(loss = loss)'
    if(is.character(loss.fun)) {
        loss.fun.string <- loss.fun # see loss()
        loss_fun <- function(x, y = out.lay)
            loss(x, y = y, type = loss.fun.string, ...)
    } else if(is.function(loss.fun)) {
        loss.fun.string <- "custom"
        loss_fun <- function(x, y = out.lay)
            loss.fun(x, y, ...)
    } else stop("'loss.fun' needs to be a character string or a function of the form function(x, y)")

    ## 4) Compile the model (compile() modifies 'model' in place)
    compile(model, optimizer = "adam", loss = loss_fun) # configure the model's learning process with the compile method

    ## Return
    structure(list(
        ## Main object
        model = model, # object of R6 class keras.engine.training.Model (directed acyclic graph of layers)
        type = "FNN", # model type
        ## Specification
        dim = dim, # integer vector of dimensions for input, hidden, output layers
        activation = activation, # character vector of activation functions for hidden and output layers
        batch.norm = batch.norm, # logical(1) indicating whether batch normalization layers are added after each hidden layer
        dropout.rate = dropout.rate, # numeric(1) specifying the fraction of input to be dropped
        n.param = nparam_FNN(model), # integer(3) giving the number of trainable, non-trainable and the total number of parameters
        ## Training
        loss.type = loss.fun.string, # character string specifying the loss function type
        n.train = NA_integer_, # integer(1) specifying the sample size for training (or NA if not trained)
        batch.size = NA_integer_, # integer(1) specifying the batch size used for training (or NA if not trained)
        n.epoch = NA_integer_, # integer(1) specifying the number of epochs used for training (or NA if not trained)
        loss = NA_real_, # numeric(n.epoch) containing the loss function values per epoch of training (or NA if not trained)
        time = system.time(NULL), # object of class "proc_time" (for training time)
        prior = matrix(, nrow = 1, ncol = dim[1])), # for a (sub-)sample of the prior (e.g. for plot())
        ## Class (part of structure())
        class = c("gnn_FNN", "gnn_GNN", "gnn_Model"))
}
