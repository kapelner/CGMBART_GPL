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

#get the Boston housing data
library(MASS)
data(Boston)
X = Boston
#X = cbind(X, rnorm(nrow(X)))
y = X$medv
X$medv = NULL

#set simulation params
Nsim = 5
ALPHA = 0.05
KnockoutPROP = c(0.01, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5, 0.9)
set_bart_machine_num_cores(4)
pct_test_data = 0.2
n_test = round(pct_test_data * nrow(X))


#bottom line metric: oos_rmse
oos_rmse_vanilla_bhd = array(NA, Nsim)
for (nsim in 1 : Nsim){
	test_indices = sample(1 : nrow(X), n_test)
	Xtest = X[test_indices, ]
	ytest = y[test_indices]
	Xtrain = X[-test_indices, ]
	ytrain = y[-test_indices]
	bart_machine = build_bart_machine(Xtrain, ytrain, verbose = FALSE, run_in_sample = FALSE)
	predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
	destroy_bart_machine(bart_machine)
	oos_rmse_vanilla_bhd[nsim] = predict_obj$rmse
	cat(".")
}
cat("\n")

############################  MCAR


oos_rmse_bhd_bartm_mcar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){	
	for (nsim in 1 : Nsim){	
		Xm = knockout_mcar(X, KnockoutPROP[i])
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		bart_machine = build_bart_machine(Xtrain, ytrain, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_bhd_bartm_mcar[i, nsim] = predict_obj$rmse
		print(oos_rmse_bhd_bartm_mcar)
	}
}


bhd_bartm_results_mcar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_bartm_mcar)
rownames(bhd_bartm_results_mcar) = c(0, KnockoutPROP)
bhd_bartm_results_mcar = cbind(bhd_bartm_results_mcar, apply(bhd_bartm_results_mcar, 1, mean))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = ALPHA / 2))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(bhd_bartm_results_mcar, "bhd_bartm_results_mcar.csv")





oos_rmse_bhd_lm_mcar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){
		Xm = knockout_mcar(X, KnockoutPROP[i])
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		Xtest = imputeMatrixByXbarj(Xtest, Xtrain)
		Xtrain = imputeMatrixByXbarj(Xtrain, Xtrain)
		Xytrain = data.frame(Xtrain , ytrain)
		lm_mod = lm(ytrain ~., data = Xytrain)
		y_hat = predict(lm_mod, newdata = Xtest)
		oos_rmse_bhd_lm_mcar[i, nsim] = sqrt((sum(ytest - y_hat)^2) / n_test) 
#		print(oos_rmse_crazy_model_lm)
	}
}


bhd_lm_results_mcar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_lm_mcar)
rownames(bhd_lm_results_mcar) = c(0, KnockoutPROP)
bhd_lm_results_mcar = cbind(bhd_lm_results_mcar, apply(bhd_lm_results_mcar, 1, mean))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = ALPHA / 2))
#mcar_results = cbind(mcar_results, apply(mcar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(bhd_lm_results_mcar, "bhd_lm_results_mcar.csv")



oos_rmse_bhd_xbarj_no_M_mcar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xm = knockout_mcar(X, KnockoutPROP[i])	
		
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		bart_machine = build_bart_machine(X = Xtrain, y = ytrain, verbose = TRUE, run_in_sample = FALSE, replace_missing_data_with_x_j_bar = TRUE, use_missing_data = FALSE)
		
		Xtest = imputeMatrixByXbarj(Xtest, Xtrain)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_bhd_xbarj_no_M_mcar[i, nsim] = predict_obj$rmse
		print(oos_rmse_bhd_xbarj_no_M_mcar)
	}
}

bhd_xbarj_no_M_results_mcar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_xbarj_no_M_mcar)
rownames(bhd_xbarj_no_M_results_mcar) = c(0, KnockoutPROP)
bhd_xbarj_no_M_results_mcar = cbind(bhd_xbarj_no_M_results_mcar, apply(bhd_xbarj_no_M_results_mcar, 1, mean))
#mcar_xbarj_no_M_results = cbind(mcar_xbarj_no_M_results, apply(mcar_xbarj_no_M_results, 1, quantile, probs = ALPHA / 2))
#mcar_xbarj_no_M_results = cbind(mcar_xbarj_no_M_results, apply(mcar_xbarj_no_M_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(bhd_xbarj_no_M_results_mcar, "bhd_xbarj_no_M_results_mcar.csv")






oos_rmse_bhd_rf_mcar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xm = knockout_mcar(X, KnockoutPROP[i])	
		
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		
		if (nrow(na.omit(Xtrain)) == nrow(Xtrain)){
			rf_mod = randomForest(x = Xtrain, y = ytrain)				
		} else {
			rf_mod = randomForest(ytrain ~ ., rfImpute(Xtrain, ytrain))		
		}
		Xtest_miss_rf = missForest(Xtest, verbose = TRUE)$ximp		
		
#		Xtest_miss_rf = imputeMatrixByXbarj(Xtest, Xtrain)
		y_hat = predict(rf_mod, Xtest_miss_rf)
		oos_rmse_bhd_rf_mcar[i, nsim] = sqrt((sum(ytest - y_hat)^2) / n_test)
		print(oos_rmse_bhd_rf_mcar)
	}
}

bhd_results_rf_mcar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_rf_mcar)
rownames(bhd_results_rf_mcar) = c(0, KnockoutPROP)
bhd_results_rf_mcar = cbind(bhd_results_rf_mcar, apply(bhd_results_rf_mcar, 1, mean))
write.csv(bhd_results_rf_mcar, "bhd_results_rf_mcar.csv")




plot(rownames(bhd_xbarj_no_M_results_mcar), bhd_xbarj_no_M_results_mcar[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], 
		type = "b", 
		main = "", 
		xlab = "Proportion Data Missing", 
		ylab = "Multiple of Baseline Error", ylim = c(1, 3),
		lwd = 3,
		col = "purple")

points(rownames(bhd_lm_results_mcar), bhd_lm_results_mcar[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], col = "red", lwd = 3, type = "b")
points(rownames(bhd_results_bartm), bhd_results_bartm[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], col = "green", lwd = 3, type = "b")
points(rownames(bhd_results_rf_mcar), bhd_results_rf_mcar[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], col = "black", lwd = 3, type = "b")





############################  MAR


oos_rmse_bhd_bartm_mar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){	
	for (nsim in 1 : Nsim){	
		Xm = knockout_mar(X, KnockoutPROP[i])
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		bart_machine = build_bart_machine(Xtrain, ytrain, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_bhd_bartm_mar[i, nsim] = predict_obj$rmse
		print(oos_rmse_bhd_bartm_mar)
	}
}


bhd_bartm_results_mar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_bartm_mar)
rownames(bhd_bartm_results_mar) = c(0, KnockoutPROP)
bhd_bartm_results_mar = cbind(bhd_bartm_results_mar, apply(bhd_bartm_results_mar, 1, mean))
#mar_results = cbind(mar_results, apply(mar_results, 1, quantile, probs = ALPHA / 2))
#mar_results = cbind(mar_results, apply(mar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(bhd_bartm_results_mar, "bhd_bartm_results_mar.csv")





oos_rmse_bhd_lm_mar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){
		Xm = knockout_mar(X, KnockoutPROP[i])
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		Xtest = imputeMatrixByXbarj(Xtest, Xtrain)
		Xtrain = imputeMatrixByXbarj(Xtrain, Xtrain)
		Xytrain = data.frame(Xtrain , ytrain)
		lm_mod = lm(ytrain ~., data = Xytrain)
		y_hat = predict(lm_mod, newdata = Xtest)
		oos_rmse_bhd_lm_mar[i, nsim] = sqrt((sum(ytest - y_hat)^2) / n_test) 
#		print(oos_rmse_crazy_model_lm)
	}
}


bhd_lm_results_mar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_lm_mar)
rownames(bhd_lm_results_mar) = c(0, KnockoutPROP)
bhd_lm_results_mar = cbind(bhd_lm_results_mar, apply(bhd_lm_results_mar, 1, mean))
#mar_results = cbind(mar_results, apply(mar_results, 1, quantile, probs = ALPHA / 2))
#mar_results = cbind(mar_results, apply(mar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(bhd_lm_results_mar, "bhd_lm_results_mar.csv")



oos_rmse_bhd_xbarj_no_M_mar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xm = knockout_mar(X, KnockoutPROP[i])	
		
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		bart_machine = build_bart_machine(X = Xtrain, y = ytrain, verbose = TRUE, run_in_sample = FALSE, replace_missing_data_with_x_j_bar = TRUE, use_missing_data = FALSE)
		
		Xtest = imputeMatrixByXbarj(Xtest, Xtrain)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_bhd_xbarj_no_M_mar[i, nsim] = predict_obj$rmse
		print(oos_rmse_bhd_xbarj_no_M_mar)
	}
}

bhd_xbarj_no_M_results_mar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_xbarj_no_M_mar)
rownames(bhd_xbarj_no_M_results_mar) = c(0, KnockoutPROP)
bhd_xbarj_no_M_results_mar = cbind(bhd_xbarj_no_M_results_mar, apply(bhd_xbarj_no_M_results_mar, 1, mean))
#mar_xbarj_no_M_results = cbind(mar_xbarj_no_M_results, apply(mar_xbarj_no_M_results, 1, quantile, probs = ALPHA / 2))
#mar_xbarj_no_M_results = cbind(mar_xbarj_no_M_results, apply(mar_xbarj_no_M_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(bhd_xbarj_no_M_results_mar, "bhd_xbarj_no_M_results_mar.csv")






oos_rmse_bhd_rf_mar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xm = knockout_mar(X, KnockoutPROP[i])	
		
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		
		if (nrow(na.omit(Xtrain)) == nrow(Xtrain)){
			rf_mod = randomForest(x = Xtrain, y = ytrain)				
		} else {
			rf_mod = randomForest(ytrain ~ ., rfImpute(Xtrain, ytrain))		
		}
		Xtest_miss_rf = missForest(Xtest, verbose = TRUE)$ximp		
		
#		Xtest_miss_rf = imputeMatrixByXbarj(Xtest, Xtrain)
		y_hat = predict(rf_mod, Xtest_miss_rf)
		oos_rmse_bhd_rf_mar[i, nsim] = sqrt((sum(ytest - y_hat)^2) / n_test)
		print(oos_rmse_bhd_rf_mar)
	}
}

bhd_rf_results_mar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_rf_mar)
rownames(bhd_rf_results_mar) = c(0, KnockoutPROP)
bhd_rf_results_mar = cbind(bhd_rf_results_mar, apply(bhd_rf_results_mar, 1, mean))
write.csv(bhd_rf_results_mar, "bhd_rf_results_mar.csv")




plot(rownames(bhd_xbarj_no_M_results_mar), bhd_xbarj_no_M_results_mar[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], 
		type = "b", 
		main = "", 
		xlab = "Proportion Data Missing", 
		ylab = "Multiple of Baseline Error", ylim = c(1, 3),
		lwd = 3,
		col = "purple")

points(rownames(bhd_lm_results_mar), bhd_lm_results_mar[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], col = "red", lwd = 3, type = "b")
points(rownames(bhd_results_bartm), bhd_results_bartm[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], col = "green", lwd = 3, type = "b")
points(rownames(bhd_rf_results_mar), bhd_rf_results_mar[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], col = "black", lwd = 3, type = "b")





############################  NMAR

oos_rmse_bhd_bartm_nmar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){	
	for (nsim in 1 : Nsim){	
		Xm = knockout_nmar(X, KnockoutPROP[i])
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		bart_machine = build_bart_machine(Xtrain, ytrain, verbose = FALSE, run_in_sample = FALSE)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_bhd_bartm_nmar[i, nsim] = predict_obj$rmse
		print(oos_rmse_bhd_bartm_nmar)
	}
}


bhd_bartm_results_nmar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_bartm_nmar)
rownames(bhd_bartm_results_nmar) = c(0, KnockoutPROP)
bhd_bartm_results_nmar = cbind(bhd_bartm_results_nmar, apply(bhd_bartm_results_nmar, 1, mean))
#nmar_results = cbind(nmar_results, apply(nmar_results, 1, quantile, probs = ALPHA / 2))
#nmar_results = cbind(nmar_results, apply(nmar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(bhd_bartm_results_nmar, "bhd_bartm_results_nmar.csv")





oos_rmse_bhd_lm_nmar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){
		Xm = knockout_nmar(X, KnockoutPROP[i])
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		Xtest = imputeMatrixByXbarj(Xtest, Xtrain)
		Xtrain = imputeMatrixByXbarj(Xtrain, Xtrain)
		Xytrain = data.frame(Xtrain , ytrain)
		lm_mod = lm(ytrain ~., data = Xytrain)
		y_hat = predict(lm_mod, newdata = Xtest)
		oos_rmse_bhd_lm_nmar[i, nsim] = sqrt((sum(ytest - y_hat)^2) / n_test) 
#		print(oos_rmse_crazy_model_lm)
	}
}


bhd_lm_results_nmar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_lm_nmar)
rownames(bhd_lm_results_nmar) = c(0, KnockoutPROP)
bhd_lm_results_nmar = cbind(bhd_lm_results_nmar, apply(bhd_lm_results_nmar, 1, mean))
#nmar_results = cbind(nmar_results, apply(nmar_results, 1, quantile, probs = ALPHA / 2))
#nmar_results = cbind(nmar_results, apply(nmar_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(bhd_lm_results_nmar, "bhd_lm_results_nmar.csv")



oos_rmse_bhd_xbarj_no_M_nmar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xm = knockout_nmar(X, KnockoutPROP[i])	
		
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		bart_machine = build_bart_machine(X = Xtrain, y = ytrain, verbose = TRUE, run_in_sample = FALSE, replace_missing_data_with_x_j_bar = TRUE, use_missing_data = FALSE)
		
		Xtest = imputeMatrixByXbarj(Xtest, Xtrain)
		predict_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
		destroy_bart_machine(bart_machine)
		oos_rmse_bhd_xbarj_no_M_nmar[i, nsim] = predict_obj$rmse
		print(oos_rmse_bhd_xbarj_no_M_nmar)
	}
}

bhd_xbarj_no_M_results_nmar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_xbarj_no_M_nmar)
rownames(bhd_xbarj_no_M_results_nmar) = c(0, KnockoutPROP)
bhd_xbarj_no_M_results_nmar = cbind(bhd_xbarj_no_M_results_nmar, apply(bhd_xbarj_no_M_results_nmar, 1, mean))
#nmar_xbarj_no_M_results = cbind(nmar_xbarj_no_M_results, apply(nmar_xbarj_no_M_results, 1, quantile, probs = ALPHA / 2))
#nmar_xbarj_no_M_results = cbind(nmar_xbarj_no_M_results, apply(nmar_xbarj_no_M_results, 1, quantile, probs = (1 - ALPHA) / 2))
write.csv(bhd_xbarj_no_M_results_nmar, "bhd_xbarj_no_M_results_nmar.csv")





oos_rmse_bhd_rf_nmar = matrix(NA, nrow = length(KnockoutPROP), ncol = Nsim)

for (i in 1 : length(KnockoutPROP)){
	for (nsim in 1 : Nsim){	
		Xm = knockout_nmar(X, KnockoutPROP[i])	
		
		test_indices = sample(1 : nrow(X), n_test)
		Xtest = Xm[test_indices, ]
		ytest = y[test_indices]
		Xtrain = Xm[-test_indices, ]
		ytrain = y[-test_indices]
		
		if (nrow(na.omit(Xtrain)) == nrow(Xtrain)){
			rf_mod = randomForest(x = Xtrain, y = ytrain)				
		} else {
			rf_mod = randomForest(ytrain ~ ., rfImpute(Xtrain, ytrain))		
		}
		Xtest_miss_rf = missForest(Xtest, verbose = TRUE)$ximp		
		
#		Xtest_miss_rf = imputeMatrixByXbarj(Xtest, Xtrain)
		y_hat = predict(rf_mod, Xtest_miss_rf)
		oos_rmse_bhd_rf_nmar[i, nsim] = sqrt((sum(ytest - y_hat)^2) / n_test)
		print(oos_rmse_bhd_rf_nmar)
	}
}

bhd_results_rf_nmar = rbind(oos_rmse_vanilla_bhd, oos_rmse_bhd_rf_nmar)
rownames(bhd_results_rf_nmar) = c(0, KnockoutPROP)
bhd_results_rf_nmar = cbind(bhd_results_rf_nmar, apply(bhd_results_rf_nmar, 1, mean))
write.csv(bhd_results_rf_nmar, "bhd_results_rf_nmar.csv")




plot(rownames(bhd_xbarj_no_M_results_nmar), bhd_xbarj_no_M_results_nmar[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], 
		type = "b", 
		main = "", 
		xlab = "Proportion Data Missing", 
		ylab = "Multiple of Baseline Error", ylim = c(1, 3),
		lwd = 3,
		col = "purple")

points(rownames(bhd_lm_results_nmar), bhd_lm_results_nmar[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], col = "red", lwd = 3, type = "b")
points(rownames(bhd_results_bartm), bhd_results_bartm[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], col = "green", lwd = 3, type = "b")
points(rownames(bhd_results_rf_nmar), bhd_results_rf_nmar[, Nsim + 1] / bhd_results_bartm[1, Nsim + 1], col = "black", lwd = 3, type = "b")

