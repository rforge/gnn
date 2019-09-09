## By Marius Hofert

## Basic tests of QMC based on GMMNs

library(gnn)

## Check (too restrictive as the OS-level TensorFlow installation will not catch
## TensorFlow installations done differently; see the stackoverflow link)
if(length(system("pip list | grep tensorflow", intern = TRUE)) > 0 && # see https://stackoverflow.com/questions/38549253/how-to-find-which-version-of-tensorflow-is-installed-in-my-system
   require(tensorflow) &&
   require(qrng) && packageVersion("qrng") >= "0.0-7" &&
   require(copula) && packageVersion("copula") >= "0.999.19")
    demo("GMMN_QMC")
