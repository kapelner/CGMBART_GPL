k_fold_cv = function(training_data, k_folds = 5, num_cores = 1, ...){
	if (k_folds <= 1 || k_folds > n){
		stop("The number of folds must be at least 2 and less than or equal to n, use \"Inf\" for leave one out")
	}
	if (k_folds == Inf){ #leave-one-out
		k_folds = bart_machine$n
	}
	
	training_data = bart_machine$training_data
	n = bart_machine$n
	
	holdout_size = round(n / k_folds)
	split_points = seq(from = 1, to = n, by = holdout_size)[1 : k_folds]
	
	L1_err = 0
	L2_err = 0
	
	for (k in 1 : k_folds){
		holdout_index_i = split_points[k]
		holdout_index_f = ifelse(k == k_folds, n, split_points[k + 1] - 1)
		print(paste("i:", holdout_index_i, "f:", holdout_index_f))
		
		test_data_k = training_data[holdout_index_i : holdout_index_f, ]
		training_data_k = training_data[-c(holdout_index_i : holdout_index_f), ]
		
		bart_machine_cv = build_bart_machine(training_data_k, ...)
		predict_obj = bart_predict_for_test_data(bart_machine_cv, test_data_k, num_cores)
		destroy_bart_machine(bart_machine_cv)
		
		#tabulate errors
		L1_err = L1_err + predict_obj$L1_err
		L2_err = L2_err + predict_obj$L2_err
	}
	
	list(L1_err = L1_err, L2_err = L2_err, rmse = sqrt(L2_err / n))
}

