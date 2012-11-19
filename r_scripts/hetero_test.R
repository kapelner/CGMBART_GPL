
directory_where_code_is = getwd() #usually we're on a linux box and we'll just navigate manually to the directory
#if we're on windows, then we're on the dev box, so use a prespecified directory
if (.Platform$OS.type == "windows"){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
}
setwd(directory_where_code_is)

source("r_scripts/bart_bakeoff.R")

library(MASS)
data(Boston)
X = Boston
colnames(X)[ncol(X)] = "y"
bart_machine = build_bart_machine(X, use_heteroskedasticity = T, num_cores = 1, debug_log = TRUE, run_in_sample = TRUE)

sigsqs = plot_sigsqs_convergence_diagnostics_hetero(bart_machine)
