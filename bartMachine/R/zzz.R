.onLoad <- function(libname, pkgname) {
  cat("Welcome to BART v1.0\n")
  assign("bartMachine_globals", new.env(), envir = parent.env(environment()))
}