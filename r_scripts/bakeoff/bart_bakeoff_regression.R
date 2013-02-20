tryCatch(library(BayesTree), error = function(e){install.packages("BayesTree")}, finally = library(BayesTree))
tryCatch(library(randomForest), error = function(e){install.packages("randomForest")}, finally = library(randomForest))
tryCatch(library(dynaTree), error = function(e){install.packages("dynaTree")}, finally = library(dynaTree))
tryCatch(library(glmnet), error = function(e){install.packages("glmnet")}, finally = library(glmnet))
tryCatch(library(gbm), error = function(e){install.packages("gbm")}, finally = library(gbm))


LAST_NAME = "kapelner"
NOT_ON_GRID = length(grep("wharton.upenn.edu", Sys.getenv(c("HOSTNAME")))) == 0


##Loads
if (NOT_ON_GRID){
	setwd(paste("C:/Users/", LAST_NAME, "/workspace/CGMBART_GPL", sep = ""))
	iter_num = 1
} else {
	setwd("../CGMBART_GPL")
}


#the following three are soon to be library(BartMachine)
source("r_scripts/bart_package.R")
source("r_scripts/bart_package_plots.R")
source("r_scripts/bart_package_validation.R")

source("r_scripts/bakeoff/create_simulated_models.R")
source("r_scripts/bakeoff/bart_bakeoff_regression_params.R")

if (NOT_ON_GRID){
	set_bart_machine_num_cores(4) #good for testing on the current machine
} else {
	set_bart_machine_num_cores(2) #most common use case for people in the real world so we want it simulated this way even if we'll lose some speed
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
	rep(real_regression_data_sets, each = run_model_N_times), 
	rep(simulated_data_sets, each = run_model_N_times)
)
#this is the last task needed in the SGE script
length(all_possible_runs)

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

#draw_boxplots_of_sim_results()
#calculate_cochran_global_pval()

calculate_cochran_global_pval = function(){
	n = nrow(avg_simulation_results)
	chi_sq = sum(-2 * log(as.numeric(avg_simulation_results[, "pval_sign_test"])))
	1 - pchisq(chi_sq, 2 * n)
}


prettify_simulation_results_and_save_as_csv = function(){
	#now update simulation results object
	rownames(simulation_results) = NULL
	simulation_results = as.data.frame(simulation_results)
	for (j in 2 : ncol(simulation_results)){
		simulation_results[, j] = as.numeric(as.character(simulation_results[, j]))
	}
	#assign it to the object
	assign("simulation_results_pretty", simulation_results, .GlobalEnv)
	#write it to file
	write.csv(simulation_results, paste("simulation_results", "/", "simulation_results.csv", sep = ""), row.names = FALSE)
}

draw_boxplots_of_sim_results = function(){
	graphics.off() #just clear it out first
	for (alpha in alphas_of_interest){
		for (beta in betas_of_interest){	
			for (num_burn_in in num_burn_ins_of_interest){
				for (num_iterations_after_burn_in in num_iterations_after_burn_ins_of_interest){
					for (num_trees in num_trees_of_interest){				
						for (data_set in c(real_regression_data_sets, simulated_data_sets)){
							draw_one_boxplot_and_save(data_set, num_trees, num_iterations_after_burn_in, num_burn_in, alpha, beta)
						}
					}
				}
			}
		}
	}
}

draw_one_boxplot_and_save = function(data_set, num_trees, num_iterations_after_burn_in, num_burn_in, alpha, beta){
	all_results = simulation_results_pretty[simulation_results_pretty$data_model == data_set & simulation_results_pretty$m == num_trees & simulation_results_pretty$N_B == num_burn_in & simulation_results_pretty$N_G == num_iterations_after_burn_in, ]
	plot_filename = paste(PLOTS_DIR, "/rmse_comp_", data_set, "_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, "_alpha_", alpha, "_beta_", beta, ".pdf", sep = "")
	pdf(file = plot_filename)
	boxplot(all_results$A_BART_rmse, all_results$R_BART_rmse, all_results$RF_rmse, 
		names = c("my BART", "Rob's BART", "RF"),
		horizontal = TRUE,
		main = paste("RMSE comparison for ", data_set, ", m = ", num_trees, ", N_B = ", num_burn_in, ", N_G = ", num_iterations_after_burn_in, " alpha = ", alpha, " beta = ", beta, sep = ""),
		xlab = paste("RMSE's (n = ", run_model_N_times, " simulations)", sep = ""))
	dev.off()	
}


avg_simulation_results_cols = c(
		"data_model", 
		"m",
		"N_B", 
		"N_G",
		"alpha", 
		"beta",
		"A_BART_rmse_avg", 
		"R_BART_rmse_avg",
		"RF_rmse_avg",
		"A_BART_rmse_se",		
		"R_BART_rmse_se",
		"A_BART_sigsq",		
		"R_BART_sigsq",	
		"A_BART_rmse_train",		
		"R_BART_rmse_train",
		"A_BART_tot_var_count",		
		"R_BART_tot_var_count",			
		"pval_sign_test",
		"A_BART_runtime",
		"R_BART_runtime",
		"RF_runtime"
)
avg_simulation_results = matrix(NA, nrow = 0, ncol = length(avg_simulation_results_cols))
colnames(avg_simulation_results) = avg_simulation_results_cols

create_avg_sim_results_and_save_as_csv = function(){
	for (alpha in alphas_of_interest){
		for (beta in betas_of_interest){	
			for (num_burn_in in num_burn_ins_of_interest){
				for (num_iterations_after_burn_in in num_iterations_after_burn_ins_of_interest){
					for (num_trees in num_trees_of_interest){				
						for (data_set in c(real_regression_data_sets, simulated_data_sets)){
							all_results_for_run = simulation_results_pretty[simulation_results_pretty$data_model == data_set & simulation_results_pretty$m == num_trees & simulation_results_pretty$N_B == num_burn_in & simulation_results_pretty$N_G == num_iterations_after_burn_in, ]
							num_a_bart_beats_r_bart = sum(all_results_for_run$A_BART_rmse > all_results_for_run$R_BART_rmse)
							pval_sign_test = binom.test(num_a_bart_beats_r_bart, run_model_N_times, 0.5)$p.value
							new_simul_row = c(
								data_set, 
								num_trees, 
								num_burn_in, 
								num_iterations_after_burn_in,
								alpha,
								beta,
								round(mean(all_results_for_run$A_BART_rmse), 2),
								round(mean(all_results_for_run$R_BART_rmse), 2),
								round(mean(all_results_for_run$RF_rmse), 1),
								round(sd(all_results_for_run$A_BART_rmse), 2),								
								round(sd(all_results_for_run$R_BART_rmse), 2),
								round(mean(all_results_for_run$A_sigsq_post_mean), 2),
								round(mean(all_results_for_run$R_sigsq_post_mean), 2),
								round(mean(all_results_for_run$A_BART_rmse_train), 2),
								round(mean(all_results_for_run$R_BART_rmse_train), 2),	
								round(mean(all_results_for_run$A_BART_tot_var_count), 2),
								round(mean(all_results_for_run$R_BART_tot_var_count), 2),								
								round(pval_sign_test, 3),
								round(all_results_for_run$A_BART_runtime, 1),
								round(all_results_for_run$R_BART_runtime, 1),
								round(all_results_for_run$RF_runtime, 1)
							)
							avg_simulation_results = rbind(avg_simulation_results, new_simul_row)					
						}
					}
				}
			}
		}
	}
	assign("avg_simulation_results", avg_simulation_results, .GlobalEnv)
	#make it pretty right away
	#now update simulation results object
	rownames(avg_simulation_results) = NULL
	avg_simulation_results = as.data.frame(avg_simulation_results)
	for (j in 2 : ncol(avg_simulation_results)){
		avg_simulation_results[, j] = as.numeric(as.character(avg_simulation_results[, j]))
	}
	#write it to file
	write.csv(avg_simulation_results, paste("simulation_results", "/", "avg_simulation_results.csv", sep = ""), row.names = FALSE)	
	assign("avg_simulation_results_pretty", avg_simulation_results, .GlobalEnv)
}

data_title = "simple_tree_structure_sigsq_half"
training_data = simulate_data_from_simulation_name(data_title)
test_data = simulate_data_from_simulation_name(data_title)




run_bart_bakeoff_iter_num(iter_num)