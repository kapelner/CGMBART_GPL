
build_bart_machine_cv = function(X = NULL, y = NULL, Xy = NULL, 
		num_burn_in = 250, 
		num_iterations_after_burn_in = 1000,
		cov_prior_vec = NULL,
		num_tree_cvs = c(200),
		k_cvs = c(2, 3, 5),
		nu_q_cvs = list(c(3, 0.9), c(3, 0.99), c(10, 0.75)),
		k_folds = 5, ...){
	
	if ((is.null(X) && is.null(Xy)) || is.null(y) && is.null(Xy)){
		stop("You need to give BART a training set either by specifying X and y or by specifying a matrix Xy which contains the response named \"y.\"\n")
	} else if (is.null(X) && is.null(y)){ #they specified Xy, so now just pull out X,y
		y = Xy$y
		Xy$y = NULL
		X = Xy
	}	
	
	min_rmse_num_tree = NULL
	min_rmse_k = NULL
	min_rmse_nu_q = NULL
	min_oos_rmse = Inf
	
	for (k in k_cvs){
		for (nu_q in nu_q_cvs){
			for (num_trees in num_tree_cvs){
				cat(paste("  BART CV try: k", k, "nu_q", paste(as.numeric(nu_q), collapse = ", "), "m", num_trees, "\n"))
				rmse = k_fold_cv(X, y, 
						k_folds = k_folds,
						num_burn_in = num_burn_in,
						num_iterations_after_burn_in = num_iterations_after_burn_in,
						cov_prior_vec = cov_prior_vec,
						num_trees = num_trees,
						k = k,
						nu = nu_q[1],
						q = nu_q[2], ...)$rmse
#				print(paste("rmse:", rmse))
				if (rmse < min_oos_rmse){
#					print(paste("new winner!"))
					min_oos_rmse = rmse					
					min_rmse_k = k
					min_rmse_nu_q = nu_q
					min_rmse_num_tree = num_trees
				}				
			}
		}
	}
	
	cat(paste("  BART CV win: k", min_rmse_k, "nu_q", paste(as.numeric(min_rmse_nu_q), collapse = ", "), "m", min_rmse_num_tree, "\n"))
	
	#now that we've found the best settings, return that bart machine
	build_bart_machine(X, y,
		num_burn_in = num_burn_in,
		num_iterations_after_burn_in = num_iterations_after_burn_in,
		cov_prior_vec = cov_prior_vec,
		num_trees = min_rmse_num_tree,
		k = min_rmse_k,
		nu = min_rmse_nu_q[1],
		q = min_rmse_nu_q[2])
}


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
		
		bart_machine_cv = build_bart_machine(training_data_k[, 1 : p], training_data_k[, (p + 1)], run_in_sample = FALSE, verbose = FALSE, ...)
		predict_obj = bart_predict_for_test_data(bart_machine_cv, test_data_k[, 1 : p], test_data_k[, (p + 1)])
		destroy_bart_machine(bart_machine_cv)
		
		#tabulate errors
		L1_err = L1_err + predict_obj$L1_err
		L2_err = L2_err + predict_obj$L2_err
	}
	cat("\n")
	
	list(L1_err = L1_err, L2_err = L2_err, rmse = sqrt(L2_err / n))
}

