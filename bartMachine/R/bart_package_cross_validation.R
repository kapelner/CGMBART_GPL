
k_fold_cv = function(X, y, k_folds = 5, ...){
	
	n = nrow(X)
	Xpreprocess = pre_process_training_data(X)
	
	p = ncol(Xpreprocess)
	
	if (k_folds <= 1 || k_folds > n){
		stop("The number of folds must be at least 2 and less than or equal to n, use \"Inf\" for leave one out")
	}
	
	
	if (k_folds == Inf){ #leave-one-out
		k_folds = n
	}	
	
	holdout_size = round(n / k_folds)
	split_points = seq(from = 1, to = n, by = holdout_size)[1 : k_folds]
	
	L1_err = 0
	L2_err = 0
	
	Xy = as.data.frame(cbind(Xpreprocess, y))
	
	for (k in 1 : k_folds){
		cat(".")
		holdout_index_i = split_points[k]
		holdout_index_f = ifelse(k == k_folds, n, split_points[k + 1] - 1)
		
		test_data_k = Xy[holdout_index_i : holdout_index_f, ]
		training_data_k = Xy[-c(holdout_index_i : holdout_index_f), ]
		
		bart_machine_cv = build_bart_machine(training_data_k[, 1 : p], training_data_k[, (p + 1)], run_in_sample = FALSE, ...)
		predict_obj = bart_predict_for_test_data(bart_machine_cv, test_data_k[, 1 : p], test_data_k[, (p + 1)])
		destroy_bart_machine(bart_machine_cv)
		
		#tabulate errors
		L1_err = L1_err + predict_obj$L1_err
		L2_err = L2_err + predict_obj$L2_err
	}
	cat("\n")
	
	list(L1_err = L1_err, L2_err = L2_err, rmse = sqrt(L2_err / n), PseudoRsq = 1 - L2_err / sum((y - mean(y))^2))
}

