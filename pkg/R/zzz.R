## Function to be executed when package is loaded (or attached)
.onLoad <- function(libname, pkgname) {
    ## Avoid startup message of layer_dense() in FNN() if TensorFlow is not
    ## built with CPU extensions such as AVX (speed improvements on CPUs)
    ## See https://stackoverflow.com/questions/65453529/suppress-hard-to-catch-output-of-an-r-function
    ## https://github.com/rstudio/tensorflow/issues/411
    ## https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
    ## https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
    ## Note: 0 = default;
    ##       1 = avoid messages;
    ##       2 = avoid messages and warnings;
    ##       3 = avoid messages, warnings and errors.
    Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "1")
    invisible()
}

