
directory_where_code_is = getwd() #usually we're on a linux box and we'll just navigate manually to the directory
#if we're on windows, then we're on the dev box, so use a prespecified directory
if (.Platform$OS.type == "windows"){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
}
setwd(directory_where_code_is)

source("r_scripts/bart_package.R")
source("r_scripts/bart_package_plots.R")
source("r_scripts/bart_package_variable_selection.R")
source("r_scripts/bart_package_f_tests.R")
source("r_scripts/missing_data/sims_functions.R")

Xy = generate_simple_model_with_missingness(500, 0, 20, gamma = 0.1)
Xytest = generate_simple_model_with_missingness(500, 0, 20, gamma = 0.1)

set_bart_machine_num_cores(4)
bart_machine = build_bart_machine(Xy = Xy,
		num_trees = 200,
		num_burn_in = 1000,
		cov_prior_vec = c(1, 10),
		num_iterations_after_burn_in = 1000,
		use_missing_data = TRUE)
bart_machine
#plot_convergence_diagnostics(bart_machine)
#plot_tree_depths(bart_machine)
#bart_machine$training_data_features
#head(bart_machine$model_matrix_training_data)
plot_y_vs_yhat(bart_machine)
plot_y_vs_yhat(bart_machine, ppis = TRUE)
X1 = as.matrix(Xytest[, 1])
colnames(X1) = colnames(Xytest)[1]
windows()
plot_y_vs_yhat(bart_machine, X = X1, y = Xytest[, 2], ppis = TRUE)

x_new = as.matrix(NA)
colnames(x_new) = "X_1"
predict_obj = bart_machine_predict(bart_machine, x_new)
hist(predict_obj$y_hat_posterior_samples, br=100)
predict_obj$y_hat






training_data = generate_simple_model_probit_with_missingness(400, mu_1 = -1, mu_2 = 1, gamma = 0.2)
Xy = training_data$Xy

set_bart_machine_num_cores(4)
bart_machine = build_bart_machine(Xy = Xy,
		num_trees = 200,
		num_burn_in = 1000,
		cov_prior_vec = c(1, 10),
		num_iterations_after_burn_in = 1000,
		use_missing_data = TRUE)
bart_machine
#plot_convergence_diagnostics(bart_machine)
#plot_tree_depths(bart_machine)
#bart_machine$training_data_features
#head(bart_machine$model_matrix_training_data)
#plot_y_vs_yhat(bart_machine)
#plot_y_vs_yhat(bart_machine, ppis = TRUE)
#X1 = as.matrix(Xytest[, 1])
#colnames(X1) = colnames(Xytest)[1]
#windows()
#plot_y_vs_yhat(bart_machine, X = X1, y = Xytest[, 2], ppis = TRUE)

x_new = as.matrix(Xy[, 1])
colnames(x_new) = "X_1"
predict_obj = bart_machine_predict(bart_machine, x_new)
#hist(predict_obj$y_hat_posterior_samples, br=100)
#predict_obj$y_hat

#bart_machine$p_hat_train
y_hat_train = ifelse(predict_obj$y_hat > 0.5, 1, 0)

windows()
plot(training_data$probs, predict_obj$y_hat, xlim = c(0,1), ylim = c(0,1))
abline(a = 0, b = 1)
#cbind(Xy$Y, y_hat_train)














################ plots of crazy model vs competitors


#set simulation params
n_dataset = 300
sigma_e = 1
missing_offset = 5


Nsim = 3
ALPHA = 0.05
KnockoutPROP = c(0.01, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5, 0.9)
set_bart_machine_num_cores(4)
pct_test_data = 0.5
n_test = round(pct_test_data * n_dataset)

Xy = generate_crazy_model(n_dataset, prop = 0, missing_offset, sigma_e)
X = Xy[, 1 : 3]
y = Xy[, 4]

#bottom line metric: oos_rmse
oos_rmse_vanilla = array(NA, Nsim)
for (nsim in 1 : Nsim){
	test_indices = sample(1 : nrow(X), n_test)
	Xtest = X[test_indices, ]
	ytest = y[test_indices]
	Xtrain = X[-test_indices, ]
	ytrain = y[-test_indices]
	bart_machine = build_bart_machine(Xtrain, ytrain, verbose = FALSE, run_in_sample = FALSE)
	predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
	destroy_bart_machine(bart_machine)
	oos_rmse_vanilla[nsim] = predict_obj$rmse
	cat(".")
}
cat("\n")


oos_rmse_crazy_model = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xy = generate_crazy_model(n_dataset, prop = KnockoutPROP[i], missing_offset, sigma_e)
		X = Xy[, 1 : 3]
		y = Xy[, 4]		
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		bart_machine = build_bart_machine(Xtrain, ytrain, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_crazy_model[i, nsim] = predict_obj$rmse
		print(oos_rmse_crazy_model)
	}
}


crazy_model_results = rbind(oos_rmse_vanilla, oos_rmse_crazy_model)
rownames(crazy_model_results) = c(0, KnockoutPROP)
crazy_model_results = cbind(crazy_model_results, apply(crazy_model_results, 1, mean))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = ALPHA / 2))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(crazy_model_results, "crazy_model_results.csv")



Xy = generate_crazy_model(n_dataset, prop = 0, missing_offset, sigma_e)
X = Xy[, 1 : 3]
y = Xy[, 4]

oos_rmse_vanilla_lm = array(NA, Nsim)
for (nsim in 1 : Nsim){
	test_indices = sample(1 : nrow(X), n_test)
	Xtest = X[test_indices, ]
	ytest = y[test_indices]
	Xtrain = X[-test_indices, ]
	ytrain = y[-test_indices]
	Xytrain = data.frame(Xtrain , ytrain)
	lm_mod = lm(ytrain ~., data = Xytrain)
	preds = predict(lm_mod, newdata = Xtest)
	oos_rmse_vanilla_lm[nsim] = sqrt((sum(ytest - preds)^2) / n_test) 
	cat(".")
}
cat("\n")


oos_rmse_crazy_model_lm = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xy = generate_crazy_model(n_dataset, prop = KnockoutPROP[i], missing_offset, sigma_e)
		X = Xy[, 1 : 3]
		y = Xy[, 4]	
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		Xtest = imputeMatrixByXbarj(Xtest, Xtrain)
		Xtrain = imputeMatrixByXbarj(Xtrain, Xtrain)
		Xytrain = data.frame(Xtrain , ytrain)
		lm_mod = lm(ytrain ~., data = Xytrain)
		preds = predict(lm_mod, newdata = Xtest)
		oos_rmse_crazy_model_lm[i, nsim] = sqrt((sum(ytest - preds)^2) / n_test) 
#		print(oos_rmse_crazy_model_lm)
	}
}


crazy_model_results_lm = rbind(oos_rmse_vanilla_lm, oos_rmse_crazy_model_lm)
rownames(crazy_model_results_lm) = c(0, KnockoutPROP)
crazy_model_results_lm = cbind(crazy_model_results_lm, apply(crazy_model_results_lm, 1, mean))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = ALPHA / 2))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(crazy_model_results_lm, "crazy_model_results_lm.csv")















#oos_rmse_crazy_model_cc = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)
#
#for (i in 1 : length(KnockoutPROP)){
#	for (nsim in 1 : Nsim){	
#		Xy = generate_crazy_model(n_dataset, prop = KnockoutPROP[i], missing_offset, sigma_e)
#		X = Xy[, 1 : 3]
#		y = Xy[, 4]		
#		test_indices = sample(1 : nrow(X), n_test)
#		
#		Xtrain = X[-test_indices, ]
#		ytrain = y[-test_indices]
#		Xtrain_ccy = na.omit(cbind(Xtrain, ytrain))
#		
#
#		
#		Xy = generate_crazy_model(n_dataset, prop = 0, missing_offset, sigma_e)
#		X = Xy[, 1 : 3]
#		y = Xy[, 4]	
#		
#		Xtest = X[test_indices, ]
#		ytest = y[test_indices]
#		Xtest_ccy = na.omit(cbind(Xtest, ytest))
#		
#		cat(nrow(Xtrain_ccy), "rows of", nrow(X), "on prop", KnockoutPROP[i], "\n")
#		if (nrow(Xtrain_ccy) == 0 || nrow(Xtest_ccy) == 0){
#			next
#		}
#		bart_machine = build_bart_machine(Xy = Xtrain_ccy, verbose = TRUE, run_in_sample = FALSE)
#		
#		Xtest = Xtest_ccy[, 1 : 3]
#		ytest = Xtest_ccy[, 4]
#		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
#		destroy_bart_machine(bart_machine)
#		oos_rmse_crazy_model_cc[i, nsim] = predict_obj$rmse
#		print(oos_rmse_crazy_model_cc)
#	}
#}
#
#crazy_model_results_cc = rbind(oos_rmse_vanilla, oos_rmse_crazy_model_cc)
#rownames(crazy_model_results_cc) = c(0, KnockoutPROP)
#crazy_model_results_cc = cbind(crazy_model_results_cc, apply(crazy_model_results_cc, 1, mean))
##mcar_results_cc = cbind(mcar_results_cc, apply(mcar_results_cc, 1, quantile, probs = ALPHA / 2))
##mcar_results_cc = cbind(mcar_results_cc, apply(mcar_results_cc, 1, quantile, probs = (1 - ALPHA) / 2))
#write.csv(crazy_model_results_cc, "crazy_model_results_cc.csv")


#oos_rmse_crazy_model_xbarj = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)
#
#for (i in 1 : length(KnockoutPROP)){
#	for (nsim in 1 : Nsim){	
#		Xy = generate_crazy_model(n_dataset, prop = KnockoutPROP[i], missing_offset, sigma_e)
#		X = Xy[, 1 : 3]
#		y = Xy[, 4]		
#		
#		test_indices = sample(1 : nrow(X), n_test)
#		Xtest = X[test_indices, ]
#		ytest = y[test_indices]
#		Xtrain = X[-test_indices, ]
#		ytrain = y[-test_indices]
#		bart_machine = build_bart_machine(X = Xtrain, y = ytrain, verbose = TRUE, run_in_sample = FALSE, replace_missing_data_with_x_j_bar = TRUE)
#		
#		Xtest = imputeMatrixByXbarj(Xtest, Xtrain)
#		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
#		destroy_bart_machine(bart_machine)
#		oos_rmse_crazy_model_xbarj[i, nsim] = predict_obj$rmse
#		print(oos_rmse_crazy_model_xbarj)
#	}
#}
#
#crazy_model_xbarj_results = rbind(oos_rmse_vanilla, oos_rmse_crazy_model_xbarj)
#rownames(crazy_model_xbarj_results) = c(0, KnockoutPROP)
#crazy_model_xbarj_results = cbind(crazy_model_xbarj_results, apply(crazy_model_xbarj_results, 1, mean))
##mcar_xbarj_results = cbind(mcar_xbarj_results, apply(mcar_xbarj_results, 1, quantile, probs = ALPHA / 2))
##mcar_xbarj_results = cbind(mcar_xbarj_results, apply(mcar_xbarj_results, 1, quantile, probs = (1 - ALPHA) / 2))
#write.csv(crazy_model_xbarj_results, "crazy_model_results_xbarj.csv")



oos_rmse_crazy_model_xbarj_no_M = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xy = generate_crazy_model(n_dataset, prop = KnockoutPROP[i], missing_offset, sigma_e)
		X = Xy[, 1 : 3]
		y = Xy[, 4]		
		
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		bart_machine = build_bart_machine(X = Xtrain, y = ytrain, verbose = TRUE, run_in_sample = FALSE, replace_missing_data_with_x_j_bar = TRUE, use_missing_data = FALSE)
		
		Xtest = imputeMatrixByXbarj(Xtest, Xtrain)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_crazy_model_xbarj_no_M[i, nsim] = predict_obj$rmse
		print(oos_rmse_crazy_model_xbarj_no_M)
	}
}

crazy_model_xbarj_no_M_results = rbind(oos_rmse_vanilla, oos_rmse_crazy_model_xbarj_no_M)
rownames(crazy_model_xbarj_no_M_results) = c(0, KnockoutPROP)
crazy_model_xbarj_no_M_results = cbind(crazy_model_xbarj_no_M_results, apply(crazy_model_xbarj_no_M_results, 1, mean))
#mcar_xbarj_no_M_results = cbind(mcar_xbarj_no_M_results, apply(mcar_xbarj_no_M_results, 1, quantile, probs = ALPHA / 2))
#mcar_xbarj_no_M_results = cbind(mcar_xbarj_no_M_results, apply(mcar_xbarj_no_M_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(crazy_model_xbarj_no_M_results, "crazy_model_results_xbarj_no_M.csv")


#plot(rownames(crazy_model_results_cc), crazy_model_results_cc[, Nsim + 1] / crazy_model_results_cc[1, Nsim + 1], 
#		type = "b", 
#		main = "MCAR error multiples", 
#		xlab = "Proportion Data MCAR", 
#		ylab = "Multiple of Error", ylim = c(0, 2),
#		lwd = 3,
#		col = "red")
plot(rownames(crazy_model_xbarj_no_M_results), crazy_model_xbarj_no_M_results[, Nsim + 1] / crazy_model_results[1, Nsim + 1], 
		type = "b", 
		main = "MCAR error multiples", 
		xlab = "Proportion Data MCAR", 
		ylab = "Multiple of Error", ylim = c(1, 3),
		lwd = 3,
		col = "purple")

points(rownames(crazy_model_results_lm), crazy_model_results_lm[, 100 + 1] / crazy_model_results[1, Nsim + 1], col = "red", lwd = 3, type = "b")


#points(rownames(crazy_model_xbarj_results), crazy_model_xbarj_results[, Nsim + 1] / crazy_model_xbarj_results[1, Nsim + 1], col = "purple", lwd = 2, type = "b")
#points(rownames(crazy_model_xbarj_no_M_results), crazy_model_xbarj_no_M_results[, Nsim + 1] / crazy_model_xbarj_no_M_results[1, Nsim + 1], col = "blue", lwd = 2, type = "b")
points(rownames(crazy_model_results), crazy_model_results[, Nsim + 1] / crazy_model_results[1, Nsim + 1], col = "green", lwd = 3, type = "b")


