### Loss functions #############################################################

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

##' @title Maximum Mean Discrepancy (MMD)
##' @param x (n, d)-tensor (for training: n = batch size, d = dimension of input
##'        dataset)
##' @param y (m, d)-tensor (for training typically m = n)
##' @param ... additional arguments passed to the underlying
##'        radial_basis_function_kernel(), most notably 'bandwidth'
##' @return 0d tensor containing the MMD
##' @author Marius Hofert
##' @note For "MMD", one has O(1/n) if x d= y and O(1) if x !d= y
MMD <- function(x, y, ...)
    tf$sqrt(    tf$reduce_mean(radial_basis_function_kernel(x, y = x, ...)) +
                tf$reduce_mean(radial_basis_function_kernel(y, y = y, ...)) -
            2 * tf$reduce_mean(radial_basis_function_kernel(x, y = y, ...))) # tf.Tensor(, shape=(), dtype=float64)

##' @title Two-sample Cramer--von Mises statistic of Remillard, Scaillet (2009,
##'        "Testing for equality between two copulas")
##' @param x (n, d)-tensor (for training: n = batch size, d = dimension of input
##'        dataset)
##' @param y (m, d)-tensor (for training typically m = n)
##' @return 0d tensor containing the two-sample CvM statistic
##' @author Marius Hofert
##' @note MWE:
##'       n <- 3
##'       m <- 4
##'       d <- 2
##'       x.R <- matrix(as.numeric(1:(n * d)), ncol = d) # dummy R object
##'       y.R <- matrix(as.numeric(1:(m * d)), ncol = d)
##'       copula::gofT2stat(x.R, y.R) # -2.404762 (res1 = 62; res2 = 133; res3 = 222)
##'       library(tensorflow)
##'       x <- tf$convert_to_tensor(x.R, dtype = "float64")
##'       y <- tf$convert_to_tensor(y.R, dtype = "float64")
##'       gnn:::CvM2(x, y) # same
CvM2 <- function(x, y)
{
    ## Idea 1: Simply convert the tensors x and y to R arrays, then call
    ##         copulas' gofT2stat() and convert result back to a tensor
    ##         via tf$convert_to_tensor(, dtype = x$dtype).
    ##         tf$convert_to_tensor(gofT2stat(as.array(x), as.array(y)), dtype = x$dtype)
    ##         => ... but this idea fails because as.array() fails due to
    ##         disabled eager execution during training; see https://stackoverflow.com/questions/66190567/how-to-convert-a-tensor-to-an-r-array-in-a-loss-function-so-without-eager-exec
    ##         => need a tensor version.

    ## Idea 2: tensorflow version but hopelessly slow:
    ## n <- nrow(x)
    ## m <- nrow(y)
    ## ## Part 1: x with x
    ## res1 <- tf$reduce_sum(tf$stack(lapply(seq_len(n), function(i) {
    ##     tf$reduce_sum(tf$stack(lapply(seq_len(n), function(k) {
    ##         tf$reduce_prod(1-tf$maximum(x[i,], x[k,]))
    ##     }))) })))
    ## ## Part 2: x with y
    ## res2 <- tf$reduce_sum(tf$stack(lapply(seq_len(n), function(i) {
    ##     tf$reduce_sum(tf$stack(lapply(seq_len(m), function(k) {
    ##         tf$reduce_prod(1-tf$maximum(x[i,], y[k,]))
    ##     }))) })))
    ## ## Part 3: y with y
    ## res3 <- tf$reduce_sum(tf$stack(lapply(seq_len(m), function(i) {
    ##     tf$reduce_sum(tf$stack(lapply(seq_len(m), function(k) {
    ##         tf$reduce_prod(1-tf$maximum(y[i,], y[k,]))
    ##     }))) })))
    ## ## Return
    ## (res1/n^2 - 2*res2/(n*m) + res3/m^2) / (1/n + 1/m) # tf.Tensor(, shape=(), dtype=float64)

    ## Idea 3: Via tf.map_fn(), but it's said to be general and rather slow, too.

    ## Idea 4: Broadcasting by expanding the dimensions; see
    ##         https://stackoverflow.com/questions/43534057/evaluate-all-pair-combinations-of-rows-of-two-tensors-in-tensorflow

    ## Part 1: x with x
    x1 <- tf$expand_dims(x, 0L) # convert (n, d)-tensor to (1, n, d)-tensor
    x2 <- tf$expand_dims(x, 1L) # convert (n, d)-tensor to (n, 1, d)-tensor
    m1 <- tf$maximum(x1, x2)
    l1 <- tf$reshape(m1, c(-1L, 2L))
    p1 <- tf$reduce_prod(1-l1, axis = 1L)
    res1 <- tf$reduce_sum(p1)

    ## Part 2: x with y
    x. <- tf$expand_dims(x, 0L) # convert (n, d)-tensor to (1, n, d)-tensor
    y. <- tf$expand_dims(y, 1L) # convert (m, d)-tensor to (m, 1, d)-tensor
    m2 <- tf$maximum(x., y.) # (m, n, d)-tensor, where [k, i, j] contains max(x[i,j], y[k,j]); check via k <- 1, i <- 2, j <- 2
    l2 <- tf$reshape(m2, c(-1L, 2L)) # (m * n, d)-tensor
    p2 <- tf$reduce_prod(1-l2, axis = 1L) # (m * n, 1)-tensor
    res2 <- tf$reduce_sum(p2) # sum all elements

    ## Part 3: y with y
    y1 <- tf$expand_dims(y, 0L) # convert (m, d)-tensor to (1, m, d)-tensor
    y2 <- tf$expand_dims(y, 1L) # convert (m, d)-tensor to (m, 1, d)-tensor
    m3 <- tf$maximum(y1, y2)
    l3 <- tf$reshape(m3, c(-1L, 2L))
    p3 <- tf$reduce_prod(1-l3, axis = 1L)
    res3 <- tf$reduce_sum(p3)

    ## Result
    n <- nrow(x)
    m <- nrow(y)
    (res1/n^2 - 2*res2/(n*m) + res3/m^2) / (1/n + 1/m) # tf.Tensor(, shape=(), dtype=float64)
}


### Main loss function #########################################################

##' @title Loss Function to Measure Statistical Discrepancy between Two Datasets
##' @param x (n, d)-tensor (for training: n = batch size, d = dimension of input
##'        dataset)
##' @param y (m, d)-tensor (for training typically m = n)
##' @param type type of reconstruction loss function. Currently available are:
##'        "MMD": (kernel) maximum mean discrepancy
##'        "CvM": Cramer-von Mises statistic of Remillard, Scaillet (2009,
##'               "Testing for equality between two copulas")
##'        "MSE": mean squared error
##'        "BCE": binary cross entropy
##' @param ... additional arguments passed to the underlying functions.
##' @return 0d tensor containing the reconstruction loss
##' @author Marius Hofert and Avinash Prasad
loss <- function(x, y, type = c("MMD", "CvM", "MSE", "BCE"), ...)
{
    type <- match.arg(type)
    switch(type,
           "MMD" = { # (theoretically) most suitable for measuring statistical discrepancy
               MMD(x, y = y, ...)
           },
           "CvM" = {
               CvM2(x, y = y)
           },
           "MSE" = { # from keras; requires nrow(x) == nrow(y)
               loss_mean_squared_error(x, y) # default for calculating the reconstruction error between two observations
           },
           "BCE" = { # from keras; requires nrow(x) == nrow(y)
               loss_binary_crossentropy(x, y) # useful for black-white images where we can interpret each pixel
           },
           stop("Wrong 'type'"))
}
