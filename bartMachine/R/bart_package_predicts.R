predict.bartMachine = function(bart_machine, new_data, prob_rule_class = NULL){
	if (bart_machine$pred_type == "regression"){	
		bart_machine_get_posterior(bart_machine, new_data)$y_hat
	} else {
		#classification
		labels = bart_machine_get_posterior(bart_machine, new_data)$y_hat > ifelse(is.null(prob_rule_class), bart_machine$prob_rule_class, prob_rule_class)
		#return whatever the raw y_levels were
		labels_to_y_levels(bart_machine, labels)
	}	
}

labels_to_y_levels = function(bart_machine, labels){
	ifelse(labels == 0, bart_machine$y_levels[1], bart_machine$y_levels[2])
}

bart_predict_for_test_data = function(bart_machine, Xtest, ytest){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	ytest_hat = predict(bart_machine, Xtest)
	
	if (bart_machine$pred_type == "regression"){
		n = nrow(Xtest)
		L2_err = sum((ytest - ytest_hat)^2)
		
		list(
				y_hat = ytest_hat,
				L1_err = sum(abs(ytest - ytest_hat)),
				L2_err = L2_err,
				rmse = sqrt(L2_err / n),
				e = ytest - ytest_hat
		)
	} else {
		confusion_matrix = as.data.frame(matrix(NA, nrow = 3, ncol = 3))
		rownames(confusion_matrix) = c(paste("actual", bart_machine$y_levels), "use errors")
		colnames(confusion_matrix) = c(paste("predicted", bart_machine$y_levels), "model errors")		
		confusion_matrix[1 : 2, 1 : 2] = as.integer(table(ytest, ytest_hat)) 
		confusion_matrix[3, 1] = round(confusion_matrix[2, 1] / (confusion_matrix[1, 1] + confusion_matrix[2, 1]), 3)
		confusion_matrix[3, 2] = round(confusion_matrix[1, 2] / (confusion_matrix[1, 2] + confusion_matrix[2, 2]), 3)
		confusion_matrix[1, 3] = round(confusion_matrix[1, 2] / (confusion_matrix[1, 1] + confusion_matrix[1, 2]), 3)
		confusion_matrix[2, 3] = round(confusion_matrix[2, 1] / (confusion_matrix[2, 1] + confusion_matrix[2, 2]), 3)
		confusion_matrix[3, 3] = round((confusion_matrix[1, 2] + confusion_matrix[2, 1]) / sum(confusion_matrix[1 : 2, 1 : 2]), 3)
		
		list(y_hat = ytest_hat, confusion_matrix = confusion_matrix)
	}

}

bart_machine_get_posterior = function(bart_machine, new_data){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	if (class(new_data) != "matrix" && class(new_data) != "data.frame"){		
		stop("X needs to be a matrix or data frame with the same column names as the training data.")
	}
#	if (sum(is.na(X)) == length(X)){
#		stop("Cannot predict on all missing data.\n")
#	}
	if (!bart_machine$use_missing_data){
		nrow_before = nrow(new_data)
		new_data = na.omit(new_data)
		if (nrow_before > nrow(new_data)){
			cat(nrow_before - nrow(new_data), "rows omitted due to missing data\n")
		}
	}
	
	if (nrow(new_data) == 0){
		stop("No rows to predict.\n")
	}
	#pull out data objects for convenience
	java_bart_machine = bart_machine$java_bart_machine
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	n = nrow(new_data)
	
	#check for errors in data
	#
	#now process and make dummies if necessary
	new_data = pre_process_new_data(new_data, bart_machine)
	
	#check for missing data if this feature was not turned on
	if (!bart_machine$use_missing_data){
		M = matrix(0, nrow = nrow(new_data), ncol = ncol(new_data))
		for (i in 1 : nrow(new_data)){
			for (j in 1 : ncol(new_data)){
				if (is.missing(new_data[i, j])){
					M[i, j] = 1
				}
			}
		}
		if (sum(M) > 0){
			cat("WARNING: missing data found in test data and BART was not built with missing data feature!\n")
		}		
	}
	
	y_hat_posterior_samples = 
		t(sapply(.jcall(bart_machine$java_bart_machine, "[[D", "getGibbsSamplesForPrediction", .jarray(new_data, dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))
	
	#to get y_hat.. just take straight mean of posterior samples, alternatively, we can let java do it if we want more bells and whistles
	y_hat = rowMeans(y_hat_posterior_samples)
	
	list(y_hat = y_hat, X = new_data, y_hat_posterior_samples = y_hat_posterior_samples)
}


calc_credible_intervals = function(bart_machine, new_data, ci_conf = 0.95){
	#first convert the rows to the correct dummies etc
	new_data = pre_process_new_data(new_data, bart_machine)
	n_test = nrow(new_data)
	
	ci_lower_bd = array(NA, n_test)
	ci_upper_bd = array(NA, n_test)	
	
	y_hat_posterior_samples = 
			t(sapply(.jcall(bart_machine$java_bart_machine, "[[D", "getGibbsSamplesForPrediction",  .jarray(new_data, dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))
	
	#to get y_hat.. just take straight mean of posterior samples, alternatively, we can let java do it if we want more bells and whistles
	y_hat = rowMeans(y_hat_posterior_samples)
	
	for (i in 1 : n_test){		
		ci_lower_bd[i] = quantile(sort(y_hat_posterior_samples[i, ]), (1 - ci_conf) / 2)
		ci_upper_bd[i] = quantile(sort(y_hat_posterior_samples[i, ]), (1 + ci_conf) / 2)
	}
	#put them together and return
	cbind(ci_lower_bd, ci_upper_bd)
}

calc_prediction_intervals = function(bart_machine, new_data, pi_conf = 0.95, normal_samples_per_gibbs_sample = 100){
	#first convert the rows to the correct dummies etc
	new_data = pre_process_new_data(new_data, bart_machine)
	n_test = nrow(new_data)
	
	pi_lower_bd = array(NA, n_test)
	pi_upper_bd = array(NA, n_test)	
	
	y_hat_posterior_samples = 
			t(sapply(.jcall(bart_machine$java_bart_machine, "[[D", "getGibbsSamplesForPrediction",  .jarray(new_data, dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
	
	
	#for each row in new_data we have to get a B x n_G matrix of draws from the normal
	
	all_prediction_samples = array(NA, c(n_test, bart_machine$num_iterations_after_burn_in, normal_samples_per_gibbs_sample))
	for (i in 1 : n_test){		
		for (n_g in 1 : bart_machine$num_iterations_after_burn_in){
			y_hat_draw = y_hat_posterior_samples[i, n_g] 
			sigsq_draw = sigsqs[n_g]
			all_prediction_samples[i, n_g, ] = rnorm(n = normal_samples_per_gibbs_sample, mean = y_hat_draw, sd = sqrt(sigsq_draw))			
		}
	}
	
	for (i in 1 : n_test){		
		pi_lower_bd[i] = quantile(c(all_prediction_samples[i,, ]), (1 - pi_conf) / 2) #fun fact: the "c" function is overloaded to vectorize an array
		pi_upper_bd[i] = quantile(c(all_prediction_samples[i,, ]), (1 + pi_conf) / 2)
	}
	#put them together and return
	cbind(pi_lower_bd, pi_upper_bd)
}