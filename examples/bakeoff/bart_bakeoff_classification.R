tryCatch(library(BayesTree), error = function(e){install.packages("BayesTree")}, finally = library(BayesTree))
tryCatch(library(randomForest), error = function(e){install.packages("randomForest")}, finally = library(randomForest))
tryCatch(library(dynaTree), error = function(e){install.packages("dynaTree")}, finally = library(dynaTree))
tryCatch(library(glmnet), error = function(e){install.packages("glmnet")}, finally = library(glmnet))
tryCatch(library(gbm), error = function(e){install.packages("gbm")}, finally = library(gbm))

#the following three are soon to be library(BartMachine)
source("r_scripts/bart_package.R")
source("r_scripts/bart_package_plots.R")
source("r_scripts/bart_package_validation.R")

#other stuff we need for the bakeoff
source("r_scripts/bakeoff/create_simulated_models.R")
source("r_scripts/bakeoff/bart_bakeoff_regression_params.R")

LAST_NAME = "kapelner"
NOT_ON_GRID = length(grep("wharton.upenn.edu", Sys.getenv(c("HOSTNAME")))) == 0


##Loads
if (NOT_ON_GRID){
	setwd(paste("C:/Users/", LAST_NAME, "/workspace/CGMBART_GPL", sep = ""))
	set_bart_machine_num_cores(4) #good for testing on the current machine
} else {
	setwd("../CGMBART_GPL")
	set_bart_machine_num_cores(2) #most common use case for people in the real world
}

#read in arguments supplied by qsub - this will tell use which gene to process
args = commandArgs(TRUE)
print(paste("args:", args))

if (length(args) > 0){
	for (i in 1 : length(args)){
		eval(parse(text = args[[i]]))
	}
}

all_possible_runs = c(
	rep(real_classification_data_sets, each = run_model_N_times), 
	rep(simulated_data_sets, each = run_model_N_times)
)

run_bart_bakeoff_iter_num = function(iter_num){

	current_run = all_possible_runs[iter_num]
		
	#now switch between real data sets and simulated data sets
	if (substr(current_run, 1, 2) == "r_"){
		raw_data = read.csv(paste("datasets//", current_run, ".csv", sep = ""))				
		#now pull out half training and half test *randomly*			
		training_indices = sort(sample(1 : nrow(raw_data), round(nrow(raw_data) * pct_train)))
		test_indices = setdiff(1 : nrow(raw_data), training_indices)
		training_data = raw_data[training_indices, ]
		test_data = raw_data[test_indices, ]
		cat(paste("starting model:", current_run, "\n"))
		results = run_models_and_save_results(training_data, test_data, current_run)		
	} else {
		raw_data = simulate_data_from_simulation_name(current_run)
		training_indices = sort(sample(1 : nrow(raw_data), round(nrow(raw_data) * pct_train)))
		test_indices = setdiff(1 : nrow(raw_data), training_indices)
		training_data = raw_data[training_indices, ]
		test_data = raw_data[test_indices, ]		
		cat(paste("starting model:", current_run, "\n"))
		results = run_models_and_save_results(training_data, test_data, current_run)		
	}
	#save results
	write.csv(results, paste("bart_bakeoff_results_", iter_num, ".csv", sep = ""), row.names = FALSE)
}

run_models_and_save_results = function(training_data, test_data, model){
	
	###set up data
	results = as.data.frame(matrix(NA, nrow = 1, ncol = length(simulation_results_cols)))
	colnames(results) = simulation_results_cols
	results$model = model
	
	p = ncol(training_data) - 1
	n_train = nrow(training_data)
	n_test = nrow(test_data)
	#split it up correctly
	Xtrain = training_data[, 1 : p]
	ytrain = training_data[, p + 1]
	Xtest = test_data[, 1 : p]
	ytest = test_data[, p + 1]
	
	##############
	# Bart Machine
	#############
	time_started = Sys.time()
	bart_machine = build_bart_machine(Xtrain, ytrain,
		num_trees = num_trees, 
		num_burn_in = num_burn_in, 
		num_iterations_after_burn_in = num_iterations_after_burn_in)

	results$A_BART_rmse_train = bart_machine$rmse_train
	
	a_bart_predictions_obj = bart_predict_for_test_data(bart_machine, Xtest, ytest)
	results$A_BART_rmse = a_bart_predictions_obj$rmse
	results$A_BART_L1 = a_bart_predictions_obj$L1_err
	results$A_sigsq_post_mean = sigsq_est(bart_machine)
	results$A_BART_tot_var_count = mean(colMeans(get_var_counts_over_chain(bart_machine)))
	
	time_finished = Sys.time()
	destroy_bart_machine(bart_machine)
	print(paste("A BART run time:", time_finished - time_started))
	
	##############
	# Bart Machine CV
	#############
	time_started = Sys.time()
	bart_machine_cv = build_bart_machine_cv(Xtrain, ytrain, 
			num_burn_in = num_burn_in, 
			num_iterations_after_burn_in = num_iterations_after_burn_in)
	
	results$A_BART_CV_rmse_train = bart_machine_cv$rmse_train
	
	a_bart_predictions_obj = bart_predict_for_test_data(bart_machine_cv, Xtest, ytest)
	results$A_BART_CV_rmse = a_bart_predictions_obj$rmse
	results$A_BART_CV_L1 = a_bart_predictions_obj$L1_err
	
	time_finished = Sys.time()
#	destroy_bart_machine(bart_machine_cv)
	print(paste("A BART CV run time:", time_finished - time_started))	
	
	##############
	# Rob BART
	#############	
	time_started = Sys.time()
	rob_bart = bart(Xtrain, ytrain,
		x.test = Xtest,
		ntree = num_trees, 
		ndpost = num_iterations_after_burn_in, 
		nskip = num_burn_in)
	
	results$R_BART_rmse_train = sqrt(sum((ytrain - rob_bart$yhat.train.mean)^2 / n_train))
	results$R_sigsq_post_mean = mean((rob_bart$sigma)^2)
	results$R_BART_tot_var_count = tryCatch({mean(apply(rob_bart$varcount, 2, mean))}, error = function(e){NA})
	
	y_hat_test = rob_bart$yhat.test.mean
	results$R_BART_L1 = sum(abs(y_hat_test - ytest))
	results$R_BART_rmse = sqrt(sum((ytest - y_hat_test)^2 / n_test))
	
	time_finished = Sys.time()
	print(paste("R BART run time:", time_finished - time_started))	

	##############
	# Random Forests
	#############		
	time_started = Sys.time()
	rf_mod = randomForest(ytrain ~ ., Xtrain)
	y_hat_test = predict(rf_mod, Xtest)
	
	results$RF_CV_rmse_train = sqrt(sum((ytrain - predict(rf_mod, training_data))^2 / n_train))
	results$RF_CV_rmse = sqrt(sum((ytest - y_hat_test)^2 / n_test))
	results$RF_CV_L1 = sum(abs(y_hat_test - ytest))
	
	time_finished = Sys.time()
	print(paste("RF run time:", time_finished - time_started))
	
	##############
	# OLS
	#############
	time_started = Sys.time()
	lm_mod = lm(ytrain ~ ., Xtrain)
	results$OLS_rmse_train = sqrt(sum((ytrain - predict(lm_mod, training_data))^2 / n_train))
	
	y_hat_test = predict(lm_mod, Xtest)
	
	
	results$OLS_rmse = sqrt(sum((ytest - y_hat_test)^2 / n_test))
	results$OLS_L1 = sum(abs(y_hat_test - ytest))
	
	time_finished = Sys.time()
	print(paste("OLS run time:", time_finished - time_started))	
	
	##############
	# Ridge Regression
	#############
	time_started = Sys.time()
	ridge_cv_mod = cv.glmnet(as.matrix(Xtrain), ytrain, alpha = 0 , nfolds = 5, type.measure = "mse")
	results$Ridge_CV_rmse_train = sqrt(sum((ytrain - predict(ridge_cv_mod, newx = as.matrix(Xtrain)))^2 / n_train))
	
	y_hat_test = predict(ridge_cv_mod, newx = as.matrix(Xtest))
		
	results$Ridge_CV_rmse = sqrt(sum((ytest - y_hat_test)^2 / n_test))
	results$Ridge_CV_L1 = sum(abs(y_hat_test - ytest))
	
	time_finished = Sys.time()
	print(paste("Ridge run time:", time_finished - time_started))	
	
	
	##############
	# Lasso Regression
	#############
	time_started = Sys.time()
	lasso_cv_mod = cv.glmnet(as.matrix(Xtrain), ytrain, alpha = 1 , nfolds = 5, type.measure = "mse")
	results$Lasso_CV_rmse_train = sqrt(sum((ytrain - predict(lasso_cv_mod, newx = as.matrix(Xtrain)))^2 / n_train))
	
	y_hat_test = predict(lasso_cv_mod, newx = as.matrix(Xtest))
	
	results$Lasso_CV_rmse = sqrt(sum((ytest - y_hat_test)^2 / n_test))
	results$Lasso_CV_L1 = sum(abs(y_hat_test - ytest))
	
	time_finished = Sys.time()
	print(paste("Lasso run time:", time_finished - time_started))	
	
	##############
	# Boosting
	#############	
	time_started = Sys.time()
	gbm_mod = gbm.fit(Xtrain, ytrain, distribution = "gaussian", interaction.depth = 4, shrinkage = 0.25)
	y_hat_train = predict(gbm_mod, Xtrain, n.tree = 100)
	results$Boosting_CV_rmse_train = sqrt(sum((ytrain - y_hat_train)^2 / n_train))
	
	y_hat_test = predict(gbm_mod, Xtest, n.tree = 100)
	
	results$Boosting_CV_rmse = sqrt(sum((ytest - y_hat_test)^2 / n_test))
	results$Boosting_CV_L1 = sum(abs(y_hat_test - ytest))
	
	time_finished = Sys.time()
	print(paste("Boosting run time:", time_finished - time_started))
	
	##############
	# DynaTree
	#############	
	time_started = Sys.time()
	dyna_tree_mod = dynaTree(Xtrain, ytrain)
	y_hat_train = predict(dyna_tree_mod, Xtrain)$mean
	results$DynaTree_rmse_train = sqrt(sum((ytrain - y_hat_train)^2 / n_train))
	
	y_hat_test = predict(dyna_tree_mod, Xtest)$mean
	
	results$DynaTree_rmse = sqrt(sum((ytest - y_hat_test)^2 / n_test))
	results$DynaTree_L1 = sum(abs(y_hat_test - ytest))
	
	time_finished = Sys.time()
	print(paste("DynaTree run time:", time_finished - time_started))	
	
	
	t(results)
	
	
	
	
	#return the results
	results
}