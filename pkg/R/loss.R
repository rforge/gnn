### Auxiliaries ################################################################

##' @title Radial Basis Function Kernel (Similarity Measure between two Samples)
##' @param x (n, d)-tensor (for training: n = batch size, d = dimension of input
##'        dataset)
##' @param y (m, d)-tensor (for training typically m = n)
##' @param bandwidth numeric containing the bandwidth parameter(s);
##'        the default seems to work well for copula type of data (the smallest
##'        value was critical for learning copulas with singular component).
##' @return (n, m)-tensor
##' @author Marius Hofert and Avinash Prasad
##' @note - This function works with tensor objects; see
##'         https://www.tensorflow.org/api_docs/python/tf and replace "tf." by
##'         "tf$" (see also https://tensorflow.rstudio.com/guide/tensorflow/tensors/).
##'         Also note that tensors are 0-indexed. For example:
##'         + tf$reduce_sum(, axis = 0L) = colSums
##'         + tf$expand_dims(arg, axis = 1L) = adds a dimension (of length/thickness 1,
##'           so no additional values) at the index 'axis' (0-indexed), so for:
##'           - arg = (n, d)-matrix => tf$expand_dims(arg, axis = 2L)
##'                 = array(arg, dim = c(nrow(arg), ncol(arg), 1))
##'           - arg = n-vector => tf$expand_dims(arg, axis = 1L)
##'                 = matrix(arg, ncol = 1)
##'       - The radial basis function kernel is a similarity measure.
##'         With the scaling factor 1/(\sqrt{2*pi} * bandwidth) one obtains
##'         the Gaussian kernel K_bandwidth(x) = dnorm(x, sd = bandwidth)
##'         used in statistics. We don't want to scale here as we don't want
##'         to blow up the value for small distances between x and y due to
##'         small bandwidths.
##'       - No checks are done for performance reasons.
##'       - Formerly used bandwidths: sqrt(c(0.001, 0.01, 0.15, 0.25, 0.50, 0.75))
##'       - MWE for how to call (outside training)
##'         n <- 3
##'         m <- 4
##'         d <- 2
##'         library(tensorflow)
##'         x <- tf$cast(matrix(1:(n * d), ncol = d), dtype = "float64") # make tensor 'double' (otherwise tf$matmul fails)
##'         y <- tf$cast(matrix(1:(m * d), ncol = d), dtype = "float64")
##'         gnn:::radial_basis_function_kernel(x, x) # (n, n)-tensor
##'         gnn:::radial_basis_function_kernel(x, y, bandwidth = c(0.1, 0.1)) # (n, m)-tensor
##'         gnn:::radial_basis_function_kernel(x, y, bandwidth = 0.1) # (n, m)-tensor
radial_basis_function_kernel <- function(x, y, bandwidth = 10^c(-3/2, -1, -1/2, -1/4, -1/8, -1/16))
{
    ## OLD
    ## dst <- tf$transpose(tf$reduce_sum(tf$square((tf$expand_dims(x, axis = 2L) -
    ##                                              tf$transpose(y))), axis = 1L))
    ## b <- tf$reshape(dst, shape = c(1L, -1L))
    ## exponent <- if(length(bandwidth) == 1) {
    ##                 tf$multiply(1 / (2 * bandwidth), b)
    ##             } else {
    ##                 tf$matmul(1 / (2 * tf$expand_dims(bandwidth, axis = 1L)), b = b)
    ##             }
    ## tf$reshape(tf$reduce_sum(tf$exp(-exponent), axis = 0L), shape = tf$shape(dst))
    x.1 <- tf$expand_dims(x, axis = 2L) # (n, d, 1)-tensor
    y.t <- tf$transpose(y) # (d, m)-tensor
    dff2 <- tf$square(x.1 - y.t) # (n, d, m)-tensor with (i, k, j) element containing (x[i,k] - y[j,k])^2
    dst2 <- tf$reduce_sum(dff2, axis = 1L) # (n, m)-matrix with (i, j) element containing sum_{k = 1}^d (x[i,k] - y[j,k])^2
    dst2.vec <- tf$reshape(dst2, shape = c(1L, -1L)) # tensor dst2 reshaped into dim (1, -1), where -1 means that the 2nd dimension is determined automatically (here: based on the first argument of (1, -1) being 1) => create one (n * m)-long row vector
    fctr.tf <- tf$convert_to_tensor(as.matrix(1 / (2 * bandwidth^2)), dtype = dst2.vec$dtype) # convert column vector (as.matrix()) to 1-column tensor (of float32/float64, as dst2.vec so that they can be multiplied without problems; note that training uses float32 whereas evaluation afterwards typically float64)
    kernels <- tf$exp(-tf$matmul(fctr.tf, b = dst2.vec)) # exp(-(x - y)^2 / (2 * h^2)); matrix multiplication of (length(bandwidth), 1)-tensor with (1, n * m)-tensor => (length(bandwidth), n * m)-tensor
    tf$reshape(tf$reduce_mean(kernels, axis = 0L), # reduce (= apply) over dimension 1 (axis = 0; cols), so compute colMeans() => (1, n * m)-tensor
               shape = tf$shape(dst2)) # reshape into the original (n, m) shape
}


### Loss function ##############################################################

##' @title Loss Function to Measure Statistical Discrepancy between Two Datasets
##' @param x (n, d)-tensor (for training: n = batch size, d = dimension of input
##'        dataset)
##' @param y (m, d)-tensor (for training typically m = n)
##' @param type type of reconstruction loss function. Currently available are:
##'        "MSE": mean squared error
##'        "binary.cross": binary cross entropy
##'        "MMD": (kernel) maximum mean discrepancy
##' @param ... additional arguments passed to the underlying functions;
##'        at the moment, this is only affects "MMD" for which "bandwidth" can
##'        be provided.
##' @return 0d tensor containing the reconstruction loss
##' @author Marius Hofert and Avinash Prasad
##' @note For "MMD", one has O(1/n) if x d= y and O(1) if x !d= y
loss <- function(x, y, type = c("MSE", "binary.cross", "MMD"), ...)
{
    type <- match.arg(type)
    switch(type,
           "MSE" = { # requires nrow(x) == nrow(y)
               loss_mean_squared_error(x, y) # default for calculating the reconstruction error between two observations
           },
           "binary.cross" = { # requires nrow(x) == nrow(y)
               loss_binary_crossentropy(x, y) # useful for black-white images where we can interpret each pixel
           },
           "MMD" = { # (theoretically) most suitable for measuring statistical discrepancy
               tf$sqrt(    tf$reduce_mean(radial_basis_function_kernel(x, y = x, ...)) +
                           tf$reduce_mean(radial_basis_function_kernel(y, y = y, ...)) -
                       2 * tf$reduce_mean(radial_basis_function_kernel(x, y = y, ...)))
           },
           stop("Wrong 'type'"))
}
