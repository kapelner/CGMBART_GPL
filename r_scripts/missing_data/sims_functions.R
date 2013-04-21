knockout_mcar = function(X, prop){
	for (i in 1 : nrow(X)){
		for (j in 1 : ncol(X)){
			if (runif(1) < prop){
				X[i, j] = NA
			}
		}
	}
	X
}


knockout_mar = function(X, prop){
	for (i in 1 : nrow(X)){
		for (j in 1 : 12){
			if (X$lstat[i] > 15){
				if (runif(1) < prop){
					X[i, j] = NA
				}
			}
			if (X$lstat[i] < 5 && j == 6){
				if (runif(1) < prop){
					X[i, j] = NA
				}				
			}
		}
	}
	X
}


knockout_nmar = function(X, prop){
	for (i in 1 : nrow(X)){
		if (X$lstat[i] < 10){
			if (runif(1) < prop){
				X$lstat[i] = NA
			}
		}
	}
	X
}


generate_crazy_model = function(n_crazy, p_crazy, prop, offset){
	Xs_crazy = matrix(runif(n_crazy * p_crazy, -1, 1), ncol = p_crazy)
	error_crazy = rnorm(n_crazy, 0, 0.01)
	X1 = Xs_crazy[, 1]
	X2 = Xs_crazy[, 2]
	X3 = Xs_crazy[, 3]
	y_crazy = Xs_crazy[, 1] + Xs_crazy[, 2] + Xs_crazy[, 3] + Xs_crazy[, 1] * Xs_crazy[, 2] + error_crazy #- Xs_crazy[, 1]^2 + Xs_crazy[, 2]^2
	
	#X1 is MCAR at 5%
#	for (i in 1 : n_crazy){
#		if (runif(1) < prop){
#			Xs_crazy[i, 1] = NA
#		}
#	}
#	
#	#X3 is MAR at 5% if X1 > 0
#	for (i in 1 : n_crazy){
#		if (runif(1) < prop && X1[i] > 0){
#			Xs_crazy[i, 3] = NA
#		}
#	}
#	
#	#X2 is NMAR at 5% if X2 > 0
#	for (i in 1 : n_crazy){
#		if (runif(1) < prop && X2[i] > 0){
#			Xs_crazy[i, 2] = NA
#		}
#	}
	
	#if X3 is missing, y bumps up by 3
	for (i in 1 : n_crazy){
		if (is.na(Xs_crazy[i, 3])){
			y_crazy[i] = y_crazy[i] + offset
		}
	}	
	
	cbind(Xs_crazy, y_crazy)
}

#k_fold_cv = function(Xmis, Xorig, y, k_folds = 5, ...){
#	
#	n = nrow(Xmis)
#	Xpreprocess = pre_process_training_data(Xmis)
#	
#	p = ncol(Xpreprocess)
#	
#	if (k_folds <= 1 || k_folds > n){
#		stop("The number of folds must be at least 2 and less than or equal to n, use \"Inf\" for leave one out")
#	}
#	
#	
#	if (k_folds == Inf){ #leave-one-out
#		k_folds = n
#	}	
#	
#	holdout_size = round(n / k_folds)
#	split_points = seq(from = 1, to = n, by = holdout_size)[1 : k_folds]
#	
#	L1_err = 0
#	L2_err = 0
#	
#	
#	for (k in 1 : k_folds){
#		cat(".")
#		holdout_index_i = split_points[k]
#		holdout_index_f = ifelse(k == k_folds, n, split_points[k + 1] - 1)
#		
#		X_test_k = Xorig[holdout_index_i : holdout_index_f, ]
#		X_train_k = Xmis[-c(holdout_index_i : holdout_index_f), ]
#		y_test_k = y[holdout_index_i : holdout_index_f]
#		y_train_k = y[-c(holdout_index_i : holdout_index_f)]
#		
#		bart_machine_cv = build_bart_machine(X_train_k, y_train_k, run_in_sample = FALSE, verbose = FALSE, ...)
#		predict_obj = bart_predict_for_test_data(bart_machine_cv, X_test_k, y_test_k)
#		destroy_bart_machine(bart_machine_cv)
#		
#		#tabulate errors
#		L1_err = L1_err + predict_obj$L1_err
#		L2_err = L2_err + predict_obj$L2_err
#	}
#	cat("\n")
#	
#	list(L1_err = L1_err, L2_err = L2_err, rmse = sqrt(L2_err / n))
#}
#
#
#
#
#
#
