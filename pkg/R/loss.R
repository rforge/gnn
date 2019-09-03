### Auxiliaries ################################################################

##' @title Matrix of Pairwise (Guassian mixture) Kernel Similarities
##' @param x 2d tensor of shape (batch size, d) (d = dimension of input dataset)
##'        => (batch size, d)-matrix
##' @param y 2d tensor of shape (batch size, d)
##' @param bandwidth numeric (vector) containing the bandwidth parameters (sigma);
##'        the default seems to work well for copula type of data.
##' @return 2d tensor of shape (batch size, batch size) => (batch size, batch size)-matrix
##' @author Marius Hofert and Avinash Prasad
##' @note - This function works with tensor objects; see https://www.tensorflow.org/guide/tensors
##'         and the R package tensorflow (for 'tf')
##'       - To get help on tensorflow functions, replace '$' by '.' and
##'         look up the corresponding Python function on https://www.tensorflow.org/api_docs/python.
##'       - This implementation is partially based on the Python code from
##'         https://github.com/tensorflow/models/blob/master/research/domain_adaptation/domain_separation/utils.py
Gaussian_mixture_kernel <- function(x, y, bandwidth = c(0.001, 0.01, 0.15, 0.25, 0.50, 0.75))
{
    dst <- tf$transpose(tf$reduce_sum(tf$square((tf$expand_dims(x, axis = 2L) -
                                                 tf$transpose(y))), axis = 1L))
    b <- tf$reshape(dst, shape = c(1L, -1L))
    exponent <- if(length(bandwidth) == 1) {
                    tf$multiply(1 / (2 * bandwidth), b)
                } else {
                    tf$matmul(1 / (2 * tf$expand_dims(bandwidth, axis = 1L)), b = b)
                }
    tf$reshape(tf$reduce_sum(tf$exp(-exponent), axis = 0L), shape = tf$shape(dst))
}


### Loss function ##############################################################

##' @title Loss Function to Measure Statistical Discrepancy between Two Datasets
##' @param x 2d tensor with shape: (batch size, d) (d = dimension of input dataset)
##' @param y 2d tensor with shape: (batch size, d)
##' @param type type of reconstruction loss function. Currently available are:
##'        "MSE": mean squared error
##'        "binary.cross": binary cross entropy
##'        "MMD": (kernel) maximum mean discrepancy
##' @param ... additional arguments passed to the underlying functions;
##'        at the moment, this is only affects "MMD" for which "bandwidth" can
##'        be provided.
##' @return 0d tensor containing the reconstruction loss
##' @author Marius Hofert and Avinash Prasad
##' @note See the R package tensorflow for 'tf'
loss <- function(x, y, type = c("MSE", "binary.cross", "MMD"), ...)
{
    type <- match.arg(type)
    switch(type,
           "MSE" = {
               loss_mean_squared_error(x, y) # default for calculating the reconstruction error between two observations
           },
           "binary.cross" = {
               loss_binary_crossentropy(x, y) # useful for black-white images where we can interpret each pixel
           },
           "MMD" = { # (theoretically) most suitable for measuring statistical discrepancy
               ## Note: We work with tensor objects here; see https://www.tensorflow.org/guide/tensors
               tf$sqrt(tf$reduce_mean(Gaussian_mixture_kernel(x, y = x, ...)) +
                       tf$reduce_mean(Gaussian_mixture_kernel(y, y = y, ...)) -
                       2 * tf$reduce_mean(Gaussian_mixture_kernel(x, y = y, ...)))
           },
           stop("Wrong 'type'"))
}
