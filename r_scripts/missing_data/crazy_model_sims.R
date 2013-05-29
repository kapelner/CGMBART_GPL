library(randomForest)
library(missForest)

directory_where_code_is = getwd() #usually we're on a linux box and we'll just navigate manually to the directory
#if we're on windows, then we're on the dev box, so use a prespecified directory
if (.Platform$OS.type == "windows"){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
}
setwd(directory_where_code_is)

source("r_scripts/bart_package.R")
source("r_scripts/bart_package_builders.R")
source("r_scripts/bart_package_plots.R")
source("r_scripts/bart_package_variable_selection.R")
source("r_scripts/bart_package_f_tests.R")
source("r_scripts/missing_data/sims_functions.R")


########## CRAZY MODEL
n_crazy = 500
p_crazy = 3
prop_missing = 0.1
offset_missing = 5
sigma_e = 1

graphics.off()

Xy = generate_crazy_model(n_crazy, p_crazy, prop_missing, offset_missing, sigma_e)
hist(Xy[, 4], br = 50, main = "distribution of response")
bart_machine = build_bart_machine(Xy = Xy, use_missing_data = TRUE, num_burn_in = 5000)
plot_y_vs_yhat(bart_machine)
windows()
plot_sigsqs_convergence_diagnostics(bart_machine)
bart_machine
check_bart_error_assumptions(bart_machine)
investigate_var_importance(bart_machine)
interaction_investigator(bart_machine, num_replicates_for_avg = 20)

###now do some predictions

###make sure it works...
xnew = as.data.frame(t(as.matrix(c(0, 0, 0))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, 0)

xnew = as.data.frame(t(as.matrix(c(1, 0, 0))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, 0)

xnew = as.data.frame(t(as.matrix(c(0, 1, 0))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, 2)

xnew = as.data.frame(t(as.matrix(c(0, 0, 1))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, 1)

xnew = as.data.frame(t(as.matrix(c(1,0,1))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, 1)

xnew = as.data.frame(t(as.matrix(c(1, 1, 1))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, 4)

xnew = as.data.frame(t(as.matrix(c(NA, 0, 0))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, -0.333)

xnew = as.data.frame(t(as.matrix(c(0, NA, 0))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, 0.8333)

xnew = as.data.frame(t(as.matrix(c(0, 0, NA))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, offset_missing)

xnew = as.data.frame(t(as.matrix(c(NA, NA, 0))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, 0.5)

xnew = as.data.frame(t(as.matrix(c(NA, NA, NA))))
pred = bart_machine_predict(bart_machine, xnew)
plot_hist_of_posterior(pred, offset_missing + 0.5)


################ plots of crazy model vs competitors


#set simulation params
n_dataset = 300
sigma_e = 1
missing_offset = 5


Nsim = 200
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


crazy_model_results_bartm = rbind(oos_rmse_vanilla, oos_rmse_crazy_model)
rownames(crazy_model_results_bartm) = c(0, KnockoutPROP)
crazy_model_results_bartm = cbind(crazy_model_results_bartm, apply(crazy_model_results_bartm, 1, mean))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = ALPHA / 2))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(crazy_model_results_bartm, "crazy_model_results_bartm.csv")



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
	y_hat = predict(lm_mod, newdata = Xtest)
	oos_rmse_vanilla_lm[nsim] = sqrt((sum(ytest - y_hat)^2) / n_test) 
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
		y_hat = predict(lm_mod, newdata = Xtest)
		oos_rmse_crazy_model_lm[i, nsim] = sqrt((sum(ytest - y_hat)^2) / n_test) 
#		print(oos_rmse_crazy_model_lm)
	}
}


crazy_model_results_lm = rbind(oos_rmse_vanilla_lm, oos_rmse_crazy_model_lm)
rownames(crazy_model_results_lm) = c(0, KnockoutPROP)
crazy_model_results_lm = cbind(crazy_model_results_lm, apply(crazy_model_results_lm, 1, mean))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = ALPHA / 2))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(crazy_model_results_lm, "crazy_model_results_lm.csv")


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



oos_rmse_crazy_model_rf = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

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
		
		if (nrow(na.omit(Xtrain)) == nrow(Xtrain)){
			rf_mod = randomForest(x = Xtrain, y = ytrain)				
		} else {
			rf_mod = randomForest(ytrain ~ ., rfImpute(Xtrain, ytrain))		
		}
		imputed = missForest(rbind(Xtest, Xtrain), verbose = TRUE)$ximp		
		Xtest_miss_rf = imputed[1 : n_test, ]
		y_hat = predict(rf_mod, Xtest_miss_rf)
		oos_rmse_crazy_model_rf[i, nsim] = sqrt((sum(ytest - y_hat)^2) / n_test)
		print(oos_rmse_crazy_model_rf)
	}
}

crazy_model_results_rf = rbind(oos_rmse_vanilla, oos_rmse_crazy_model_rf)
rownames(crazy_model_results_rf) = c(0, KnockoutPROP)
crazy_model_results_rf = cbind(crazy_model_results_rf, apply(crazy_model_results_rf, 1, mean))
write.csv(crazy_model_results_rf, "crazy_model_results_rf.csv")


oos_rmse_crazy_model_bart_with_imp = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i_knockout in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xy = generate_crazy_model(n_crazy, prop = KnockoutPROP[i_knockout], offset_missing, sigma_e)
		X = Xy[, 1 : 3]
		y = Xy[, 4]		
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		bart_machine = build_bart_machine(Xtrain, ytrain, verbose = FALSE, run_in_sample = FALSE, add_imputations = TRUE, debug_log = TRUE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_crazy_model_bart_with_imp[i_knockout, nsim] = predict_obj$rmse
		print(oos_rmse_crazy_model_bart_with_imp)
	}
}

crazy_model_results_bart_with_imp = rbind(oos_rmse_vanilla, oos_rmse_crazy_model_bart_with_imp)
rownames(crazy_model_results_bart_with_imp) = c(0, KnockoutPROP)
crazy_model_results_bart_with_imp = cbind(crazy_model_results_bart_with_imp, apply(crazy_model_results_bart_with_imp, 1, mean))
write.csv(crazy_model_results_bart_with_imp, "crazy_model_results_bart_with_imp.csv")


plot(rownames(crazy_model_xbarj_no_M_results), crazy_model_xbarj_no_M_results[, Nsim + 1] / crazy_model_results_bartm[1, Nsim + 1], 
	type = "b", 
	main = "", 
	xlab = "Proportion Data Missing", 
	ylab = "Multiple of Baseline Error", ylim = c(1, 3),
	lwd = 3,
	col = "purple")

points(rownames(crazy_model_results_lm), crazy_model_results_lm[, Nsim + 1] / crazy_model_results_bartm[1, Nsim + 1], col = "red", lwd = 3, type = "b")
points(rownames(crazy_model_results_bartm), crazy_model_results_bartm[, Nsim + 1] / crazy_model_results_bartm[1, Nsim + 1], col = "green", lwd = 3, type = "b")
points(rownames(crazy_model_results_rf), crazy_model_results_rf[, Nsim + 1] / crazy_model_results_bartm[1, Nsim + 1], col = "black", lwd = 3, type = "b")
points(rownames(crazy_model_results_bart_with_imp), crazy_model_results_bart_with_imp[, Nsim + 1] / crazy_model_results_bartm[1, Nsim + 1], col = "brown", lwd = 3, type = "b")






