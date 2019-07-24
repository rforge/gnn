## By Marius Hofert

## Basic tests of dataset-handling functions

library(gnn)

myobj1 <- 1:10
myobj2 <- 10:1
file <- "foo1.rda"
filename <- rm_ext(file)
save(myobj1, myobj2, file = file)
rm(myobj1, myobj2)

## Testing exists_rda()
stopifnot(exists_rda(file)) # check existence of file 'file' with...
stopifnot(exists_rda(filename)) # ... and without file ending '.rda'
stopifnot(exists_rda(file, objnames = "myobj1")) # check existence of 'myobj1' inside 'file' with...
stopifnot(exists_rda(filename, objnames = "myobj1")) # ... and without file ending '.rda'
stopifnot(exists_rda("SP500_const", package = "qrmdata")) # check existence of file named 'SP500_const' in 'qrmdata'
stopifnot(exists_rda("SP500_const", objnames = "SP500_const_info", package = "qrmdata")) # check existence of object 'SP500_const_info' inside 'SP500_const' in 'qrmdata'

## Testing read_rda()
stopifnot(names(read_rda(c("myobj1", "myobj2"), file = file)) == c("myobj1", "myobj2"))
foo <- read_rda("SP500_const_info", file = "SP500_const", package = "qrmdata")
stopifnot(is.data.frame(foo))

## Testing save_rda()
save_rda(foo, file = "foo2.rda", names = "SP500info2")
bar <- read_rda("SP500info2", file = "foo2.rda")
stopifnot(identical(bar, foo))

## Clean-up
stopifnot(file.remove(file, "foo2.rda"))
