options(repos = "http://lib.stat.cmu.edu/R/CRAN")

tryCatch(library(randomForest), error = function(e){install.packages("randomForest")}, finally = library(randomForest))
tryCatch(library(rpart), error = function(e){install.packages("rpart")}, finally = library(rpart))
tryCatch(library(xtable), error = function(e){install.packages("xtable")}, finally = library(xtable))
tryCatch(library(BayesTree), error = function(e){install.packages("BayesTree")}, finally = library(BayesTree))


run_other_model_and_plot_y_vs_yhat = function(y_hat, 
		model_name, 
		test_data, 
		X, y,
		extra_text = NULL, 
		data_title = "data_model", 
		save_plot = FALSE,
		bart_machine = NULL, 
		sigsqs = NULL,
		avg_num_splits_by_vars = NULL,
		y_hat_train = NULL,
		runtime = NULL,
		create_plot = FALSE){
	L1_err = sum(abs(test_data$y - y_hat))
	L2_err = sum((test_data$y - y_hat)^2)	
	rmse = sqrt(L2_err / length(y_hat))
	L2_err_train = sum((y - y_hat_train)^2)
	rmse_train = sqrt(L2_err_train / length(y_hat_train))
	
	if (create_plot){
		if (save_plot){
			save_plot_function(bart_machine, paste("yvyhat_", model_name, sep = ""), data_title)
		}	
		else {
			dev.new()
		}		
		plot(test_data$y, 
				y_hat, 
				main = paste("y/yhat ", model_name, " model L1/2 = ", round(L1_err, 1), "/", round(L2_err, 1), " rmse = ", round(rmse, 2), ifelse(is.null(extra_text), "", paste("\n", extra_text)), sep = ""), 
				xlab = "y", 
				ylab = "y_hat")	
		if (save_plot){	
			dev.off()
		}
	}
	
	list(y_hat = y_hat, 
			L1_err = L1_err, 
			L2_err = L2_err, 
			rmse = rmse, 
			sigsqs = sigsqs,
			avg_num_splits_by_vars = avg_num_splits_by_vars,
			rmse_train = rmse_train,
			runtime = runtime)		
}



run_random_forests_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	before = Sys.time()
	rf_mod = randomForest(y ~., training_data)
	y_hat = predict(rf_mod, test_data)
	after = Sys.time()
	print(paste("RF run time:", after - before))	
	run_other_model_and_plot_y_vs_yhat(y_hat, 
			"RF", 
			test_data, 
			training_data, 
			extra_text, 
			data_title, 
			save_plot, 
			bart_machine,
			runtime = after - before)
}

run_bayes_tree_bart_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	p = ncol(training_data) - 1
	before = Sys.time()
	bayes_tree_bart_mod = bart(x.train = training_data[, 1 : p],
			y.train = training_data$y,
			sigest = sd(training_data$y), #we force the sigma estimate to be the std dev of y
			x.test = test_data[, 1 : p],
			sigdf = 3, #$\nu = 3$, this is the same value we used in the implementation
			sigquant = 0.9, 
			k = 2, #same as we have it
			power = DEFAULT_BETA, #same as we have it
			base = DEFAULT_ALPHA, #same as we have it
			ntree = bart_machine$num_trees, #keep it the same -- default is 200 in BayesTree... interesting...
			ndpost = bart_machine$num_iterations_after_burn_in, #keep it the same
			nskip = bart_machine$num_burn_in, #keep it the same --- default is 100 in BayesTree -- huh??
			usequants = TRUE, #this is a tiny bit different...check with Ed
			numcut = length(training_data$y), #this is a tiny bit different...check with Ed
			verbose = TRUE)
	
#	out = list(yhat = bayes_tree_bart_mod$yhat.test.mean, sigmas = bayes_tree_bart_mod$sigma)
	y_hat = bayes_tree_bart_mod$yhat.test.mean
	after = Sys.time()
	print(paste("R BART run time:", after - before))	
	sigsqs = (bayes_tree_bart_mod$sigma)^2
	avg_num_splits_by_vars = tryCatch({apply(bayes_tree_bart_mod$varcount, 2, mean)}, error = function(e){NA})
	y_hat_train = bayes_tree_bart_mod$yhat.train.mean
	run_other_model_and_plot_y_vs_yhat(y_hat, 
			"R_BART", 
			test_data, 
			training_data,
			extra_text, 
			data_title, 
			save_plot, 
			bart_machine, 
			sigsqs = sigsqs, 
			avg_num_splits_by_vars = avg_num_splits_by_vars,
			y_hat_train = y_hat_train,
			runtime = after - before)
}

run_cart_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	before = Sys.time()
	cart_model = rpart(y ~ ., training_data)
	y_hat = predict(cart_model, test_data)
	after = Sys.time()
	print(paste("CART run time:", after - before))		
	run_other_model_and_plot_y_vs_yhat(y_hat, 
			"CART", 
			test_data, 
			training_data, 
			extra_text, 
			data_title, 
			save_plot, 
			bart_machine,
			runtime = after - before)
}


save_plot_function = function(bart_machine, identifying_text, data_title){
	if (is.null(bart_machine)){
		stop("you cannot save a plot unless you pass the bart_machine object", call. = FALSE)
	}
	num_iterations_after_burn_in = bart_machine[["num_iterations_after_burn_in"]]
	num_burn_in = bart_machine$num_burn_in
	num_trees = bart_machine$num_trees	
	alpha = bart_machine$alpha
	beta = bart_machine$beta
	plot_filename = paste(PLOTS_DIR, "/", data_title, "_", identifying_text, "_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, "_alpha_", alpha, "_beta_", beta, ".pdf", sep = "")
	tryCatch({pdf(file = plot_filename)}, error = function(e){})
	append_to_log(paste("plot saved as", plot_filename))
}