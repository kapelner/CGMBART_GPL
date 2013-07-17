predict.bartMachine = function(bart_machine, new_data){
	bart_machine_predict(bart_machine, new_data)$y_hat
}

bart_predict_for_test_data = function(bart_machine, Xtest, ytest){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	ytest_hat = predict(bart_machine, Xtest)
	n = nrow(Xtest)
	L2_err = sum((ytest - ytest_hat)^2)
	
	list(
		y_hat = ytest_hat,
		L1_err = sum(abs(ytest - ytest_hat)),
		L2_err = L2_err,
		rmse = sqrt(L2_err / n),
		e = ytest - ytest_hat
	)
}



bart_machine_predict = function(bart_machine, new_data, ppi = 0.95){
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
	
	ppi_a = apply(y_hat_posterior_samples, 1, quantile, probs = (1 - ppi) / 2)
	ppi_b = apply(y_hat_posterior_samples, 1, quantile, probs = ppi + (1 - ppi) / 2)
	
	list(y_hat = y_hat, X = new_data, y_hat_posterior_samples = y_hat_posterior_samples, ppi_a = ppi_a, ppi_b = ppi_b)
}


calc_ppis_from_prediction = function(bart_machine, new_data, ppi_conf = 0.95){
	#first convert the rows to the correct dummies etc
	new_data = pre_process_new_data(new_data, bart_machine)
	n_test = nrow(new_data)
	
	ppi_lower_bd = array(NA, n_test)
	ppi_upper_bd = array(NA, n_test)	
	
	y_hat_posterior_samples = 
			t(sapply(.jcall(bart_machine$java_bart_machine, "[[D", "getGibbsSamplesForPrediction",  .jarray(new_data, dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))
	
	#to get y_hat.. just take straight mean of posterior samples, alternatively, we can let java do it if we want more bells and whistles
	y_hat = rowMeans(y_hat_posterior_samples)
	
	for (i in 1 : n_test){		
		ppi_lower_bd[i] = quantile(sort(y_hat_posterior_samples[i, ]), (1 - ppi_conf) / 2)
		ppi_upper_bd[i] = quantile(sort(y_hat_posterior_samples[i, ]), (1 + ppi_conf) / 2)
	}
	#put them together and return
	cbind(ppi_lower_bd, ppi_upper_bd)
}