
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

#get the Boston housing data
library(MASS)
data(Boston)
X = Boston
#X = cbind(X, rnorm(nrow(X)))
y = X$medv
X$medv = NULL

#set simulation params
Nsim = 10
K_FOLDS = 5
KnockoutPROP = c(0.01, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5, 0.9)
set_bart_machine_num_cores(4)
pct_test_data = 0.2
n_test = round(pct_test_data * nrow(X))

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


########### MCAR

oos_rmse_mcar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 7 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		Xtrainmis = knockout_mcar(Xtrain, KnockoutPROP[i])
		bart_machine = build_bart_machine(Xtrainmis, ytrain, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_mcar[i, nsim] = predict_obj$rmse
		print(oos_rmse_mcar)
	}
}


mcar_results = rbind(oos_rmse_vanilla, oos_rmse_mcar)
rownames(mcar_results) = c(0, KnockoutPROP)
mcar_results = cbind(mcar_results, apply(mcar_results, 1, mean))
plot(rownames(mcar_results), mcar_results[, 4] / mcar_results[1, 4], 
		type = "b", 
		main = "MCAR error multiples", 
		xlab = "Proportion Data MCAR", 
		ylab = "RMSE relative to full")
write.csv(mcar_results, "mcar_results.csv")


oos_rmse_mcar_cc = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		Xtrainmis = knockout_mcar(Xtrain, KnockoutPROP[i])
		Xtrainmisccy = na.omit(cbind(Xtrainmis, ytrain))
		cat(nrow(Xtrainmisccy), "rows of", nrow(X), "on prop", KnockoutPROP[i], "\n")
		if (nrow(Xtrainmisccy) == 0){
			next
		}
		bart_machine = build_bart_machine(Xy = Xtrainmisccy, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_mcar_cc[i, nsim] = predict_obj$rmse
		print(oos_rmse_mcar_cc)
	}
}

mcar_results_cc = rbind(oos_rmse_vanilla, oos_rmse_mcar_cc)
rownames(mcar_results_cc) = c(0, KnockoutPROP)
mcar_results_cc = cbind(mcar_results_cc, apply(mcar_results_cc, 1, mean))

plot(rownames(mcar_results_cc), mcar_results_cc[, 4] / mcar_results_cc[1, 4], 
		type = "b", 
		main = "MCAR error multiples", 
		xlab = "Proportion Data MCAR", 
		ylab = "Multiple of Error", 
		col = "red")
points(rownames(mcar_results), mcar_results[, 4] / mcar_results[1, 4], col = "green", type = "b")
write.csv(mcar_results_cc, "mcar_results_cc.csv")


########### MAR


oos_rmse_mar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		Xtrainmis = knockout_mar(Xtrain, KnockoutPROP[i])
		bart_machine = build_bart_machine(Xtrainmis, ytrain, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_mar[i, nsim] = predict_obj$rmse
		print(oos_rmse_mar)
	}
}

mar_results = rbind(oos_rmse_vanilla, oos_rmse_mar)
rownames(mar_results) = c(0, KnockoutPROP)
mcr_results = cbind(mar_results, apply(mar_results, 1, mean))
plot(rownames(mar_results), mar_results[, 4] / mar_results[1, 4], 
		type = "b", 
		main = "MAR error multiples", 
		xlab = "Proportion Data MAR", 
		ylab = "RMSE relative to full")
write.csv(mar_results, "mar_results.csv")



oos_rmse_mar_cc = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		Xtrainmis = knockout_mar(Xtrain, KnockoutPROP[i])
		Xtrainmisccy = na.omit(cbind(Xtrainmis, ytrain))
		cat(nrow(Xtrainmisccy), "rows of", nrow(X), "on prop", KnockoutPROP[i], "\n")
		if (nrow(Xtrainmisccy) == 0){
			next
		}		
		bart_machine = build_bart_machine(Xy = Xtrainmisccy, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_mar_cc[i, nsim] = predict_obj$rmse
		print(oos_rmse_mar_cc)
	}
}

mar_results_cc = rbind(oos_rmse_vanilla, oos_rmse_mar_cc)
rownames(mar_results_cc) = c(0, KnockoutPROP)
mcr_results = cbind(mar_results_cc, apply(mar_results_cc, 1, mean))
plot(rownames(mar_results_cc), mar_results_cc[, 4] / mar_results_cc[1, 4], 
		type = "b", 
		main = "MAR error multiples", 
		xlab = "Proportion Data MAR", 
		ylab = "Multiple of Error", 
		col = "red")
points(rownames(mar_results), mar_results[, 4] / mar_results[1, 4], col = "green", type = "b")
write.csv(mar_results_cc, "mar_results_cc.csv")



########### NMAR

oos_rmse_nmar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		Xtrainmis = knockout_nmar(Xtrain, KnockoutPROP[i])
		bart_machine = build_bart_machine(Xtrainmis, ytrain, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_nmar[i, nsim] = predict_obj$rmse
		print(oos_rmse_nmar)
	}
}

nmar_results = rbind(oos_rmse_vanilla, oos_rmse_nmar)
rownames(nmar_results) = c(0, KnockoutPROP)
mcr_results = cbind(nmar_results, apply(nmar_results, 1, mean))
plot(rownames(nmar_results), nmar_results[, 4] / nmar_results[1, 4], 
		type = "b", 
		main = "NMAR error multiples", 
		xlab = "Proportion Data NMAR", 
		ylab = "RMSE relative to full")
write.csv(nmar_results, "nmar_results.csv")



oos_rmse_nmar_cc = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = X[test_indices, ]
		ytest = y[test_indices]
		Xtrain = X[-test_indices, ]
		ytrain = y[-test_indices]
		Xtrainmis = knockout_nmar(Xtrain, KnockoutPROP[i])
		Xtrainmisccy = na.omit(cbind(Xtrainmis, ytrain))
		cat(nrow(Xtrainmisccy), "rows of", nrow(X), "on prop", KnockoutPROP[i], "\n")
		if (nrow(Xtrainmisccy) == 0){
			next
		}		
		bart_machine = build_bart_machine(Xy = Xtrainmisccy, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_nmar_cc[i, nsim] = predict_obj$rmse
		print(oos_rmse_nmar_cc)
	}
}

nmar_results_cc = rbind(oos_rmse_vanilla, oos_rmse_nmar_cc)
rownames(nmar_results_cc) = c(0, KnockoutPROP)
mcr_results = cbind(nmar_results_cc, apply(nmar_results_cc, 1, mean))
plot(rownames(nmar_results_cc), nmar_results_cc[, 4] / nmar_results_cc[1, 4], 
		type = "b", 
		main = "NMAR error multiples", 
		xlab = "Proportion Data NMAR", 
		ylab = "Multiple of Error", 
		col = "red")
points(rownames(nmar_results), nmar_results[, 4] / nmar_results[1, 4], col = "green", type = "b")
write.csv(nmar_results_cc, "nmar_results_cc.csv")






########## CRAZY MODEL
n_crazy = 500
p_crazy = 3
prop_missing = 0.0
offset_missing = 3

graphics.off()
Xy = generate_crazy_model(n_crazy, p_crazy, prop_missing, offset_missing)
hist(Xy[, 4], br = 50, main = "distribution of response")
bart_machine = build_bart_machine_cv(X = X[, 1:3], y = Xy[, 4], use_missing_data = TRUE, num_burn_in = 1000)
plot_y_vs_yhat(bart_machine)
windows()
plot_sigsqs_convergence_diagnostics(bart_machine)
bart_machine
check_bart_error_assumptions(bart_machine)
investigate_var_importance(bart_machine)
interaction_investigator(bart_machine, num_replicates_for_avg = 20)

###now do some predictions
pred = bart_machine_predict(bart_machine, c(0, 0, 0)) #E[Y] = 0
hist(pred$y_hat_posterior_samples, br = 50, main = paste("posterior of yhat, mean =", mean(pred$y_hat_posterior_samples)))
abline(v = 0, col = "blue", lwd = 2)

pred = bart_machine_predict(bart_machine, c(1, 1, 1)) #E[Y] = 0
hist(pred$y_hat_posterior_samples, br = 50, main = paste("posterior of yhat, mean =", mean(pred$y_hat_posterior_samples)))
abline(v = 0, col = "blue", lwd = 2)


pred = bart_machine_predict(bart_machine, c(NA, 0, 0)) #E[Y] = 0
hist(pred$y_hat_posterior_samples, br = 50, main = paste("posterior of yhat, mean =", mean(pred$y_hat_posterior_samples)))
abline(v = 0, col = "blue", lwd = 2)

pred = bart_machine_predict(bart_machine, c(0, NA, 0)) #E[Y] = 0
hist(pred$y_hat_posterior_samples, br = 50, main = paste("posterior of yhat, mean =", mean(pred$y_hat_posterior_samples)))
abline(v = 0, col = "blue", lwd = 2)

pred = bart_machine_predict(bart_machine, c(0, 0, NA)) #E[Y] = 3
hist(pred$y_hat_posterior_samples, br = 50, main = paste("posterior of yhat, mean =", mean(pred$y_hat_posterior_samples)))
abline(v = 3, col = "blue", lwd = 2)

pred = bart_machine_predict(bart_machine, c(NA, NA, 0)) #E[Y] = 0
hist(pred$y_hat_posterior_samples, br = 50, main = paste("posterior of yhat, mean =", mean(pred$y_hat_posterior_samples)))
abline(v = 0, col = "blue", lwd = 2)

pred = bart_machine_predict(bart_machine, c(NA, NA, NA)) #E[Y] = 3
hist(pred$y_hat_posterior_samples, br = 50, main = paste("posterior of yhat, mean =", mean(pred$y_hat_posterior_samples)))
abline(v = 3, col = "blue", lwd = 2)

pred = bart_machine_predict(bart_machine, c(1, 0, 0)) #E[Y] = 0
hist(pred$y_hat_posterior_samples, br = 50, main = paste("posterior of yhat, mean =", mean(pred$y_hat_posterior_samples)))
abline(v = 1, col = "blue", lwd = 2)

pred = bart_machine_predict(bart_machine, c(0, 1, 0)) #E[Y] = 1
hist(pred$y_hat_posterior_samples, br = 50, main = paste("posterior of yhat, mean =", mean(pred$y_hat_posterior_samples)))
abline(v = 1, col = "blue", lwd = 1)







#get oos-rmse
#oos_rmse_vanilla = array(NA, Nsim)
#for (nsim in 1 : Nsim){
#	error_obj = k_fold_cv(X, X, y, k_folds = K_FOLDS, verbose = FALSE)
#	oos_rmse_vanilla[nsim] = error_obj$rmse
#}
#
##### DO MCAR
#
#
#
#oos_rmse_mcar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)
#
#for (i in 1 : length(KnockoutPROP)){
#	for (nsim in 1 : Nsim){	
#		Xmis = knockout_mcar(X, KnockoutPROP[i])
#		error_obj = k_fold_cv(Xmis, X, y, k_folds = K_FOLDS, verbose = FALSE)
#		oos_rmse_mcar[i, nsim] = error_obj$rmse
#		print(oos_rmse_mcar)
#	}
#}
#
#mcar_results = rbind(oos_rmse_vanilla, oos_rmse_mcar)
#rownames(mcar_results) = c(0, KnockoutPROP)
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, mean))
#
##plot(rownames(mcar_results), mcar_results[, 1])
##points(rownames(mcar_results), mcar_results[, 2])
##points(rownames(mcar_results), mcar_results[, 3])
##plot(rownames(mcar_results), mcar_results[, 4])
#plot(rownames(mcar_results), mcar_results[, 4] / mcar_results[1, 4], type = "b", main = "MCAR error multiples", xlab = "Proportion Data MCAR", ylab = "Multiple Error")
#
#
#oos_rmse_mcar_cc = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)
#
#for (i in 1 : length(KnockoutPROP)){
#	for (nsim in 1 : Nsim){	
#		Xmis = knockout_mcar(X, KnockoutPROP[i])
#		Xmisccy = na.omit(cbind(Xmis, y))
#		cat(nrow(Xmisccy), "rows of", nrow(X), "on prop", KnockoutPROP[i], "\n")
#		if (nrow(Xmisccy) == 0){
#			next
#		}
#		error_obj = k_fold_cv(Xmisccy[, 1 : ncol(X)], Xmisccy[, 1 : ncol(X)], Xmisccy[, (ncol(X) + 1)], k_folds = K_FOLDS, use_missing_data = FALSE)
#		oos_rmse_mcar_cc[i, nsim] = error_obj$rmse
#		print(oos_rmse_mcar_cc)
#	}
#}
#
#mcar_results_cc = rbind(oos_rmse_vanilla, oos_rmse_mcar_cc)
#rownames(mcar_results_cc) = c(0, KnockoutPROP)
#mcar_results_cc = cbind(mcar_results_cc, apply(mcar_results_cc, 1, mean))
#
#plot(rownames(mcar_results_cc), mcar_results_cc[, 4] / mcar_results_cc[1, 4], type = "b", main = "MCAR cc error multiples", xlab = "Proportion Data MCAR", ylab = "Multiple of Error", col = "red")
#points(rownames(mcar_results), mcar_results[, 4] / mcar_results[1, 4], col = "green", type = "b")
#
##plot(rownames(mcar_results), mcar_results[, 4] / mcar_results[1, 4], type = "b", main = "MCAR error multiples", xlab = "Proportion Data MCAR", ylab = "Multiple Error")
##
##lm_data = as.data.frame(cbind(as.numeric(rownames(mcar_results)), mcar_results[, 4]))
##
##mod = lm(lm_data[,2] ~ lm_data[,1]) #poly(as.numeric(rownames(mcar_results)), 3)
##summary(mod)
##abline(mod)
##predict.lm(mod, data.frame(c(0.02, 0.04))) #seq(0,1,0.01)
#
#par(mfrow = c(4,4))
#for (j in 1 : ncol(X)){
#	hist(X[, j], br = 50, main = colnames(X)[j], xlab = "")
#}
#
##build MAR model
#
#
#
#oos_rmse_mar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)
#
#for (i in 1 : length(KnockoutPROP)){
#	for (nsim in 1 : Nsim){	
#		Xmis = knockout_mar(X, KnockoutPROP[i])
#		error_obj = k_fold_cv(Xmis, X, y, k_folds = K_FOLDS)
#		oos_rmse_mar[i, nsim] = error_obj$rmse
#		print(oos_rmse_mar)
#	}
#}
#
#
#mar_results = rbind(oos_rmse_vanilla, oos_rmse_mar)
#rownames(mar_results) = c(0, KnockoutPROP)
#mar_results = cbind(mar_results, apply(mar_results, 1, mean))

