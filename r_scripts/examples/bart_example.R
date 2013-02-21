
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
y = X$medv
X$medv = NULL
#X$chas = as.character(X$chas)
#X$rad = as.factor(X$rad)

#split it into test and training
Xtrain = X[1 : (nrow(X) / 2), ]
ytrain = y[1 : (nrow(X) / 2)]
Xtest = X[(nrow(X) / 2 + 1) : nrow(X), ]
ytest = y[(nrow(X) / 2 + 1) : nrow(X)]
#build the BART machine
#bart_machines = list()
#for (i in 1 : 5000){
#graphics.off()
#windows()

set_bart_machine_num_cores(4)
bart_machine = build_bart_machine(Xtrain, ytrain,
		num_trees = 200,
		num_burn_in = 300,
		num_iterations_after_burn_in = 2000, nu =3.4, debug_log=TRUE)


bart_machine = build_bart_machine_cv(Xtrain, ytrain,
		num_burn_in = 300,
		num_iterations_after_burn_in = 2000,
		debug_log = TRUE)


	
	cat(paste("built bart machine #", i, "\n"))
#}
summary(bart_machine)
bart_machine$training_data_features

plot_y_vs_yhat(bart_machine, ppis=T)
plot_y_vs_yhat(bart_machine, X = Xtest, y = ytest, ppis = TRUE)

#convenience to predict on the test data automatically computing SSE, etc
predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
predict_obj$rmse


library(BayesTree)
rob = bart(x.train = Xtrain,y.train=ytrain,x.test=Xtest,ndpost=2000,nskip=500)
sqrt(sum((rob$yhat.test.mean - ytest)^2) / length(ytest))
mean(rob$sigma)


interaction_investigator(bart_machine, num_replicates_for_avg = 5, num_var_plot = 20, num_trees_bottleneck = 20)
investigate_var_importance(bart_machine)
investigate_var_importance(bart_machine, type = "trees")

plot_y_vs_yhat(bart_machine, ppis=T)

plot_tree_num_nodes(bart_machine)
plot_tree_depths(bart_machine)
plot_mh_acceptance_reject(bart_machine)
plot_convergence_diagnostics(bart_machine)


plot_sigsqs_convergence_diagnostics(bart_machine)
hist_sigsqs(bart_machine)
check_bart_error_assumptions(bart_machine)



pd_plot(bart_machine, 6)


rmses = array(NA, 20)
for (k in 2 : 20){
	rmse = k_fold_cv(X, k_folds = k, 
		num_trees = 200,
		num_burn_in = 250, 
		num_iterations_after_burn_in = 1000)$rmse
	rmses[k] = rmse
}
plot(1:20, rmses, type="l")
k = 1:20
summary(lm(rmses ~ k))


#get PPIs for test data
ppi_obj = calc_ppis_from_prediction(bart_machine, Xtest)



#now test the variable importance
##generate the Friedman data
n = 2000
p = 100
X = as.data.frame(matrix(runif(n * p, 0 , 1), ncol = p))
error = rnorm(n, 0, 1)

y = 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5] + error
#
#Xy = data.frame(X, y)
#summary(lm(y ~ ., Xy))
#Xy = read.csv("datasets/r_friedman_hd.csv")
#Xtrain = Xy[1 : 80, 1 : 100]
#ytrain = Xy[1 : 80, 101]
#Xtest = Xy[81 : 100, 1 : 100]
#ytest = Xy[81 : 100, 101]

Xtrain = X[1 : 1000,]
ytrain = y[1 : 1000]
Xtest = X[1001 : 2000, ]
ytest = y[1001 : 2000]

set_bart_machine_num_cores(1)
bart_machine = build_bart_machine(Xtrain, ytrain,
	num_trees = 200,
	num_burn_in = 600, 
	num_iterations_after_burn_in = 2000,
	debug_log = T)

plot_y_vs_yhat(bart_machine, ppis=T)
plot_y_vs_yhat(bart_machine, X = Xtest, y = ytest, ppis = TRUE)

summary(bart_machine)
var_importance_by_shuffling(bart_machine, num_var_plot = 20)
var_importance_by_dropping_variable(bart_machine, num_var_plot = 20)
hist_sigsqs(bart_machine)
windows()
plot_sigsqs_convergence_diagnostics(bart_machine)
check_bart_error_assumptions(bart_machine)
windows()
plot_y_vs_yhat(bart_machine, ppis = T)
rmse_by_num_trees(bart_machine)
rmse_by_num_trees(bart_machine, in_sample = TRUE)

windows()
investigate_var_importance(bart_machine, num_replicates_for_avg = 5,num_var_plot = 15, type = "trees")
windows()
interaction_investigator(bart_machine, num_replicates_for_avg = 5, num_var_plot = 15)

library(BayesTree)
rob = bart(x.train = Xy[, 1:100], y.train = Xy[ ,101], ndpost = 1000, nskip = 1000, sigest = sd(Xy[,101]), ntree = 200)
counts = apply(rob$varcount,2,sum)
rob$sigest
mean(rob$sigma)
counts/sum(counts)
names(counts)=colnames(Xy)[1:100]
sort(counts, dec = T)
interaction_investigator(bart_machine, num_replicates_for_avg = 5, num_var_plot=20)




#X = read.csv("datasets/r_boston.csv")
Xtrain = X[1 : (nrow(X) / 2), ]
Xtest = X[(nrow(X) / 2 + 1) : nrow(X), ]
bart_machine = build_bart_machine(X, 
		num_trees = 200,
		num_burn_in = 2000, 
		num_iterations_after_burn_in = 500)
summary(bart_machine)
plot_y_vs_yhat(bart_machine)

pred_obj = bart_predict_for_test_data(bart_machine_cv, Xtest)
pred_obj$rmse




library(randomForest)
rf = randomForest(x = Xtrain[, 1 : 13], y = Xtrain$y, importance = TRUE)
importance(rf, type=1)
preds = predict(rf,newdata = Xtest[, 1 : 13])
sqrt(sum((preds - Xtest$y)^2) / length(preds))




source("r_scripts/create_simulated_models.R")

Xy = simulate_data_from_simulation_name("bivariate_linear")
#Xy = read.csv("datasets/r_just_noise.csv")
#Xy=Xy[1:500,]

bart_machine = build_bart_machine(Xy[, 1 : (ncol(Xy) - 1)], Xy$y,
		num_trees = 200,
		num_burn_in = 1000, 
		num_iterations_after_burn_in = 500,
		s_sq_y = "mse")
summary(bart_machine)
hist_sigsqs(bart_machine)
plot_sigsqs_convergence_diagnostics(bart_machine)
rmse_by_num_trees(bart_machine)

bart_machine = build_bart_machine(Xy[, 1 : (ncol(Xy) - 1)], rnorm(nrow(Xy)),
		num_trees = 200,
		num_burn_in = 1000, 
		num_iterations_after_burn_in = 500,
		s_sq_y = "mse")

library(BayesTree)
rob = bart(x.train = Xy[, 1:100], y.train = Xy[ , 101], ndpost = 500, nskip = 1000, ntree = 20, sigest = sd(Xy[,101]))
counts = apply(rob$varcount,2,sum)

sort_props= sort(counts/sum(counts),dec = T)
sort_props
barplot(sort_props)
rob$sigest
mean(rob$sigma)




#PROBIT BART
Xy = read.csv("datasets/c_breastcancer.csv", header = FALSE)
X = Xy[, 1 : (ncol(Xy) - 1)]
y = as.factor(Xy[, ncol(Xy)])

set_bart_machine_num_cores(4)
bart_machine = build_bart_machine(X, y,
		num_trees = 200,
		num_burn_in = 1000, 
		num_iterations_after_burn_in = 500)
bart_machine
