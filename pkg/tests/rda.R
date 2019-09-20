## By Marius Hofert

## Basic tests of dataset-handling functions

library(gnn)

myobj1 <- 1:10
myobj2 <- 10:1
file1. <- tempfile("foo1", fileext = ".rda") # tempfile() for CRAN
file1 <- rm_ext(file1.)
save(myobj1, myobj2, file = file1.)
rm(myobj1, myobj2)

## Testing exists_rda()
stopifnot(exists_rda(file1.)) # check existence of file 'file' with...
stopifnot(exists_rda(file1)) # ... and without file ending '.rda'
stopifnot(exists_rda(file1., names = "myobj1")) # check existence of 'myobj1' inside 'file' with...
stopifnot(exists_rda(file1,  names = "myobj1")) # ... and without file ending '.rda'
stopifnot(exists_rda("SP500_const", package = "qrmdata")) # check existence of file named 'SP500_const' in 'qrmdata'
stopifnot(exists_rda("SP500_const", names = "SP500_const_info", package = "qrmdata")) # check existence of object 'SP500_const_info' inside 'SP500_const' in 'qrmdata'

## Testing read_rda()
stopifnot(names(read_rda(c("myobj1", "myobj2"), file = file1.)) == c("myobj1", "myobj2"))
foo <- read_rda("SP500_const", names = "SP500_const_info", package = "qrmdata")
stopifnot(is.data.frame(foo))

## Testing save_rda()
file2. <- tempfile("foo2", fileext = ".rda") # for CRAN
save_rda(foo, file = file2., names = "SP500info2")
bar <- read_rda(file2., names = "SP500info2")
stopifnot(identical(bar, foo))

## Clean-up
stopifnot(file.remove(file1., file2.))
