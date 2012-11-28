
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
X$medv = log(X$medv)
colnames(X)[ncol(X)] = "y"

#split it in half
Xtrain = X[1 : (nrow(X) / 2), ]
Xtest = X[(nrow(X) / 2 + 1) : nrow(X), ]


bart_machine = build_bart_machine(Xtrain, 
				run_in_sample = TRUE, 
				num_trees = 10, 
				debug_log = TRUE, 
				num_burn_in = 1000, 
				num_iterations_after_burn_in = 1000, 
				num_cores = 3)

for (i in 1 : 100){
	print(i)
	predict_obj = bart_predict(bart_machine, Xtest, num_cores = 1)
	predict_obj = bart_predict_for_test_data(bart_machine, Xtest, num_cores = 1)
}
ppi_obj = calc_ppis_from_prediction(predict_obj)
#bart_machine = build_bart_machine(X, use_heteroskedasticity = F, num_cores = 10, run_in_sample = TRUE)
#
#sigsqs = plot_sigsqs_convergence_diagnostics_hetero(bart_machine)		
