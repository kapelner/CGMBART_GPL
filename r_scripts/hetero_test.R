
source("r_scripts/bart_bakeoff.R")

library(MASS)
data(Boston)
X=Boston
colnames(X)[ncol(X)] = "y"
bart_machine=build_bart_machine(X, use_heteroskedasticity=T)

sigsqs = plot_sigsqs_convergence_diagnostics_hetero(bart_machine)
