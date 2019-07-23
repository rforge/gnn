### Auxiliary functions for checking existence, loading and saving data ########

##' @title Checking whether Datasets(s) Exist
##' @param file character string (with or without ending .rda) specifying the
##'        name of the file considered
##' @param objnames names of objects to be checked for existence
##' @param package package name in which to check or NULL (the default) in
##'        which case the current working directory is checked
##' @return logical
##' @author Marius Hofert
##' @note For .rds: file.exists(file)
exists_rda <- function(file, objnames, package = NULL)
{
    stopifnot(is.character(file), length(file) == 1)
    if(hasArg(objnames)) { # check existence of object 'objnames' inside 'file'
        ## Note: data() per default load()s the objects into .GlobalEnv
        ##       and thus overwrites existing objects with the same name.
        ##       To avoid this we create a new, temporary environment
        ##       and load the data there.
        myenvir <- new.env() # new environment (in order to not overwrite .GlobalEnv entries)
        if(is.null(package)) {
            if(!file.exists(file))
                stop("File '",file,"' does not exist.")
            load(file, envir = myenvir)
            ## Alternatively, could work with attach()
        } else {
            file <- file_path_sans_ext(basename(file)) # data() fails if '.rda' is included
            data(list = file, package = package, envir = myenvir) # load the .rda into 'myenvir'
        }
        res <- objnames %in% ls(, envir = myenvir) # now check whether objects 'objnames' exist inside the .rda
        rm(myenvir) # clean-up
        res # return
    } else { # check existence of 'file' as an .rda
        if(is.null(package)) {
            file.exists(file)
        } else {
            file %in% data(package = package, envir = .GlobalEnv)[["results"]][,"Item"] # 'envir' is only a dummy here to avoid a CRAN note
        }
    }

}

##' @title Reading Objects from an .rda from the Current Package or File in the
##'        Current Working Directory
##' @param objnames names of objects to be read
##' @param file character string (with or without ending .rda) specifying
##'        the file to read from
##' @param package package name from which to load the objects or NULL (the default)
##'        in which case the current working directory is searched.
##' @return the read object(s)
##' @author Marius Hofert
##' @note For .rds: readRDS()
read_rda <- function(objnames, file, package = NULL)
{
    stopifnot(is.character(file), length(file) == 1)
    if(!all(exists_rda(file, objnames = objnames, package = package)))
        stop("Not all objects specified by 'objnames' exist in file '",file,"'")
    myenvir <- new.env()
    if(is.null(package)) {
        load(file, envir = myenvir) # load .rda into myenvir
        ## Alternatively, could work with attach()
    } else {
        file <- file_path_sans_ext(basename(file)) # data() fails if '.rda' is included
        data(list = file, package = package, # in the current package
             envir = myenvir) # loads objects in 'file' into 'myenvir'
    }
    res <- if(length(objnames) == 1) {
               get(objnames, envir = myenvir) # get object(s) from 'myenvir'
           } else {
               mget(objnames, envir = myenvir)
           }
    rm(myenvir) # clean-up
    res # return
}

##' @title Saving Objects under a given Name in a Given .rda File
##' @param ... objects to be saved in 'file' under names specified by 'names'
##' @param file character string (with or without ending '.rda') specifying
##'        the file to save to
##' @param names character vector of names under which the objects are saved
##'        in 'file'
##' @return nothing (generates an .rda by side-effect)
##' @author Marius Hofert
##' @note For .rds: saveRDS()
save_rda <- function(..., file, names)
{
    stopifnot(is.character(file), length(file) == 1)
    args <- list(...)
    len <- length(args)
    stopifnot(length(names) == len)
    for(i in seq_len(len))
        assign(names[i], value = args[[i]]) # name the objects in 'args' as specified by 'names'
    save(list = names, file = file) # save R objects in 'file' under the provided 'names'
}

##' @title Reading, Renaming and Saving an .rda Object
##' @param oldname character string specifying the object to be read
##' @param oldfile file name (with or without ending .rda) specifying from which
##'        the object named 'oldname' is read
##' @param newname character string specifying the new name under which the object
##'        is to be saved
##' @param newfile file name (with ending .rda) specifying where the object named
##'        'oldname' is saved under the name 'newname'
##' @param package see ?read_rda
##' @return nothing (generates an .rda by side-effect)
##' @author Marius Hofert
##' @note An .rds can simply be renamed (as no object name is stored)
rename_rda <- function(oldname, oldfile = paste0(oldname, collapse = "_"),
                       newname, newfile = paste0(oldfile,".rda"), package = NULL)
{
    dat <- read_rda(oldname, file = oldfile, package = package)
    save_rda(dat, file = newfile, names = newname) # does not save 'dat' as 'dat' but under the right name 'newname'
}
