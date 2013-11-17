.onLoad <- function(libname, pkgname) {
  cat("Welcome to BART v1.0\n")
  assign("pkg_globals", new.env(), envir = parent.env(environment()))
  assign("VERSION", "1.0b", pkg_globals)
  assign("JAR_DEPENDENCIES", c("bart_java.jar", "commons-math-2.1.jar", "trove-3.0.3.jar", "junit-4.10.jar"), pkg_globals)
}