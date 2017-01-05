.onLoad <- function(libname, pkgname) {
   op <- options()
   op.fastknn <- list(
      
   )
   toset <- !(names(op.fastknn) %in% names(op))
   
   if (any(toset))
      options(op.fastknn[toset])
   
   invisible()
}

.onAttach <- function(libname, pkgname) {
   packageStartupMessage("FastKNN version 0.9.0")
}