.onLoad <- function(libname, pkgname) {
  .jpackage(pkgname, lib.loc = libname)
  cat("Welcome to BART v1.0")
}