### Auxiliary functions for checking existence, loading and saving data ########

##' @title Checking whether Datasets(s) Exist
##' @param file character string (with or without extension .rda) specifying the
##'        name of the file considered
##' @param names names of objects to be checked for existence
##' @param package name of the package in which to check; if NULL (the default)
##'        the current working directory is used.
##' @return logical
##' @author Marius Hofert
##' @note - For .rds: file.exists(file)
##'       - File extensions can largely mess this up:
##'         + file.exists() needs the file extension .rda to find the file
##'         + data() must have no extension
exists_rda <- function(file, names, package = NULL)
{
    stopifnot(is.character(file), length(file) == 1)
    file  <- rm_ext(file) # remove file extension
    file. <- paste0(file,".rda") # file with extension
    if(hasArg(names)) { # need to check existence of object 'names' inside 'file.'
        ## names <- rm_ext(basename(names)) # remove path and file extension (as '%in% ls()' fails if '.rda' is included)
        ## Note (2019-10-06): Commented out the last line as it can remove parts of objects names
        ##                    which are then not found in 'file' anymore.
        ## Note: data() per default load()s the objects into .GlobalEnv
        ##       and thus overwrites existing objects with the same name.
        ##       To avoid this we create a new, temporary environment
        ##       and load the data there.
        myenvir <- new.env() # new environment (in order to not overwrite .GlobalEnv entries)
        if(is.null(package)) {
            if(!file.exists(file.)) return(rep(FALSE, length(names))) # if file does not exist, 'names' can't exist in 'file.'
            load(file., envir = myenvir) # alternatively, could work with attach()
        } else {
            data(list = file, package = package, envir = myenvir) # load the .rda into 'myenvir'; must have no extension
        }
        res <- names %in% ls(, envir = myenvir) # now check whether objects 'names' exist inside the .rda
        rm(myenvir) # clean-up
        res # return
    } else { # check existence of 'file.' as an .rda
        if(is.null(package)) {
            file.exists(file.) # needs extension
        } else {
            file %in% data(package = package, envir = .GlobalEnv)[["results"]][,"Item"] # 'envir' is only a dummy here to avoid a CRAN note; needs no extension
        }
    }
}

##' @title Reading Objects from an .rda from the Current Package or File in the
##'        Current Working Directory
##' @param file character string (with or without extension .rda) specifying
##'        the file to read from
##' @param names character vector of names of objects to be read
##' @param package name of the package from which to load the objects; if NULL
##'        (the default) the current working directory is searched.
##' @return the read object(s)
##' @author Marius Hofert
##' @note For .rds: readRDS()
read_rda <- function(file, names, package = NULL)
{
    stopifnot(is.character(file))
    file  <- rm_ext(file) # remove file extension
    file. <- paste0(file,".rda") # file with extension
    names <- if(hasArg(names)) {
                 ## rm_ext(basename(names)) # otherwise get() fails below
                 ## Note (2019-10-06): Commented out the last line as we don't want to change names
                 names
             } else {
                 paste0(file, collapse = "_")
                 ## Note (2019-10-06): 'file' in the last line was rm_ext(basename(file.))
             }
    if(!all(exists_rda(file., names = names, package = package)))
        if(is.null(package)) {
            stop("Not all objects specified by 'names' exist in file '",file.,"' in the local working directory")
        } else {
            stop("Not all objects specified by 'names' exist in file '",file.,"' in package '",package,"'")
        }
    myenvir <- new.env()
    if(is.null(package)) {
        if(!file.exists(file.)) # needs extension
            stop("File '",file.,"' does not exist.")
        load(file., envir = myenvir) # load .rda into myenvir
        ## Alternatively, could work with attach()
    } else {
        data(list = file, package = package, # in the current package
             envir = myenvir) # loads objects in 'file' into 'myenvir'; must have no extension
    }
    res <- if(length(names) == 1) {
               get(names, envir = myenvir) # get object(s) from 'myenvir'
           } else {
               mget(names, envir = myenvir)
           }
    rm(myenvir) # clean-up
    res # return
}

##' @title Saving Objects under a given Name in a Given .rda File
##' @param ... objects to be saved in 'file' under names specified by 'names'
##' @param file character string (with or without extension '.rda') specifying
##'        the file to save to
##' @param names character vector of names under which the objects are saved
##'        in 'file'
##' @return nothing (generates an .rda by side-effect)
##' @author Marius Hofert
##' @note For .rds: saveRDS()
save_rda <- function(..., file, names = NULL)
{
    stopifnot(is.character(file), length(file) == 1)
    args <- list(...)
    len <- length(args)
    if(is.null(names)) {
        nms <- deparse(substitute(list(...))) # get names of provided arguments
        nms <- substring(nms, first = 6, last = nchar(nms) - 1) # strip away "list(" and ")"
        names <- unlist(strsplit(nms, split = ", "))
    }
    stopifnot(length(names) == len)
    for(i in seq_len(len))
        assign(names[i], value = args[[i]]) # name the objects in 'args' as specified by 'names'
    save(list = names, file = file) # save R objects in 'file' under the provided 'names'
}

##' @title Reading, Renaming and Saving an .rda Object
##' @param oldname character string specifying the object to be read
##' @param oldfile file name (with or without extension .rda) specifying from which
##'        the object named 'oldname' is read
##' @param newname character string (without extension .rda) specifying the new name
##'        under which the object is to be saved
##' @param newfile file name (with extension .rda) specifying where the object named
##'        'oldname' is saved under the name 'newname'
##' @param package see ?read_rda
##' @return nothing (generates an .rda by side-effect)
##' @author Marius Hofert
##' @note An .rds can simply be renamed (as no object name is stored)
rename_rda <- function(oldname, oldfile = paste0(oldname, collapse = "_"),
                       newname, newfile = paste0(newname, collapse = "_", ".rda"),
                       package = NULL)
{
    dat <- read_rda(oldname, file = oldfile, package = package)
    save_rda(dat, file = newfile, names = newname) # does not save 'dat' as 'dat' but under the right name 'newname'
}
