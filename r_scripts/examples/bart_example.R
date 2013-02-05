
directory_where_code_is = getwd() #usually we're on a linux box and we'll just navigate manually to the directory
#if we're on windows, then we're on the dev box, so use a prespecified directory
if (.Platform$OS.type == "windows"){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
}
setwd(directory_where_code_is)

source("r_scripts/bart_package.R")
source("r_scripts/bart_package_plots.R")

#get some data
library(MASS)
data(Boston)
X = Boston
X$medv = log(X$medv)
#X$chas = as.character(X$chas)
colnames(X)[ncol(X)] = "y"

#split it into test and training
Xtrain = X[1 : (nrow(X) / 2), ]
Xtest = X[(nrow(X) / 2 + 1) : nrow(X), ]

#build the BART machine
#bart_machines = list()
#for (i in 1 : 5000){
graphics.off()
	bart_machine = build_bart_machine(Xtrain, 
		num_trees = 200,
		num_burn_in = 250, 
		num_iterations_after_burn_in = 1000,
		num_cores = 4)
	
	cat(paste("built bart machine #", i, "\n"))
#}
summary(bart_machine)
interaction_investigator(bart_machine)
investigate_var_importance(bart_machine)

plot_y_vs_yhat(bart_machine)

plot_tree_num_nodes(bart_machine)
plot_tree_depths(bart_machine)
plot_mh_acceptance_reject(bart_machine)
plot_convergence_diagnostics(bart_machine)


plot_sigsqs_convergence_diagnostics(bart_machine)
check_bart_error_assumptions(bart_machine)

#convenience to predict on the test data automatically computing SSE, etc
predict_obj = bart_predict_for_test_data(bart_machine, Xtest, num_cores = 4)

rmses = array(NA, 20)
for (k in 2 : 20){
	rmse = k_fold_cv(X, k_folds = k, 
		num_trees = 200,
		num_burn_in = 250, 
		num_iterations_after_burn_in = 1000,
		num_cores = 3)$rmse
	rmses[k] = rmse
}
plot(1:20, rmses, type="l")
k = 1:20
summary(lm(rmses ~ k))


#get PPIs for test data
ppi_obj = calc_ppis_from_prediction(bart_machine, Xtest)

library(randomForest)
rf = randomForest(x = Xtrain[,1:13], y =Xtrain[,14])
preds = predict(rf,newdata = Xtest[,1:13])
sqrt(sum((preds-Xtest[,14])^2)/length(preds))

#now test the variable importance
#generate the Friedman data
n = 500
x1 = runif(n, 0, 1)
x2 = runif(n, 0, 1)
x3 = runif(n, 0, 1)
x4 = runif(n, 0, 1)
x5 = runif(n, 0, 1)
x6 = runif(n, 0, 1)
x7 = runif(n, 0, 1)
x8 = runif(n, 0, 1)
x9 = runif(n, 0, 1)
x10 = runif(n, 0, 1)
error = rnorm(n, 0, 10)

y = 1-sin(pi * x1 * x2) + 20 * (x3 - 0.5)^2 + 10 * x4 + 5 * x5 + error

Xy = data.frame(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y)
summary(lm(y ~ ., Xy))

bart_machine = build_bart_machine(Xy, 
	run_in_sample = TRUE, 
	num_trees = 200,
	num_burn_in = 1000, 
	num_iterations_after_burn_in = 1000, 
	num_cores = 4)

check_bart_error_assumptions(bart_machine)

for (var in 1 : 10){
	varsign = get_variable_significance(bart_machine, var, num_cores = 3)
	print(var)
	print(varsign$p_value)
}
