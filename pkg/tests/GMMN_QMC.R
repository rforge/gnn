## By Marius Hofert

## Basic tests of QMC based on GMMNs

library(gnn)
if(require(tensorflow) &&
   packageVersion("qrng") >= "0.0-7" &&
   packageVersion("copula") >= "0.999.19")
    demo("GMMN_QMC")