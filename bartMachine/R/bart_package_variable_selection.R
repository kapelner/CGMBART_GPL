
var_selection_by_permute_response_three_methods = function(bart_machine, num_reps_for_avg = 10, num_permute_samples = 100, num_trees_for_permute = 20, alpha = 0.05, plot = TRUE, num_var_plot = Inf, bottom_margin = 10){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	
	permute_mat = matrix(NA, nrow = num_permute_samples, ncol = bart_machine$p)
	colnames(permute_mat) = bart_machine$training_data_features_with_missing_features
	
	cat("avg")
	var_true_props_avg = get_averaged_true_var_props(bart_machine, num_reps_for_avg, num_trees_for_permute)
	
	#now sort from high to low
	var_true_props_avg = sort(var_true_props_avg, decreasing = TRUE)
	
	cat("null")
	for (b in 1 : num_permute_samples){
		permute_mat[b, ] = get_null_permute_var_importances(bart_machine, num_trees_for_permute)
	}
	cat("\n")
	
	#sort permute mat
	permute_mat = permute_mat[, names(var_true_props_avg)]
	
	pointwise_cutoffs = apply(permute_mat, 2, quantile, probs = 1 - alpha)
	important_vars_pointwise_names = names(var_true_props_avg[var_true_props_avg > pointwise_cutoffs & var_true_props_avg > 0])
	important_vars_pointwise_col_nums = sapply(1 : length(important_vars_pointwise_names), function(x){which(important_vars_pointwise_names[x] == bart_machine$training_data_features_with_missing_features)})
	
	max_cut = quantile(apply(permute_mat, 1 ,max), 1 - alpha)
	important_vars_simul_max_names = names(var_true_props_avg[var_true_props_avg >= max_cut & var_true_props_avg > 0])	
	important_vars_simul_max_col_nums = sapply(1 : length(important_vars_simul_max_names), function(x){which(important_vars_simul_max_names[x] == bart_machine$training_data_features_with_missing_features)})
	
	
	perm_se = apply(permute_mat, 2, sd)
	perm_mean = apply(permute_mat, 2, mean)
	cover_constant = bisectK(tol = .01 , coverage = 1 - alpha, permute_mat = permute_mat, x_left = 1, x_right = 20, countLimit = 100, perm_mean = perm_mean, perm_se = perm_se)
	important_vars_simul_se_names = names(var_true_props_avg[which(var_true_props_avg >= perm_mean + cover_constant * perm_se & var_true_props_avg > 0)])	
	important_vars_simul_se_col_nums = sapply(1 : length(important_vars_simul_se_names), function(x){which(important_vars_simul_se_names[x] == bart_machine$training_data_features_with_missing_features)})
	
	
	if (plot){
		par(mar = c(bottom_margin, 6, 3, 0))
		if (num_var_plot == Inf | num_var_plot > bart_machine$p){
			num_var_plot = bart_machine$p
		}
		
		par(mfrow = c(2, 1))
		##pointwise plot
		plot(1 : num_var_plot, var_true_props_avg[1 : num_var_plot], type = "n", xlab = NA, xaxt = "n", ylim = c(0, max(max(var_true_props_avg), max_cut * 1.1)),
				main = "Local Procedure", ylab = "proportion included")
		axis(1, at = 1 : num_var_plot, labels = names(var_true_props_avg[1 : num_var_plot]), las = 2)
		for (j in 1 : num_var_plot){
			points(j, var_true_props_avg[j], pch = ifelse(var_true_props_avg[j] <= quantile(permute_mat[, j], 1 - alpha), 1, 16))
		}
		
		sapply(1 : num_var_plot, function(s){segments(s, 0, x1 = s, quantile(permute_mat[, s], 1 - alpha), col = "forestgreen")})
		
		##simul plots
		plot(1 : num_var_plot, var_true_props_avg[1 : num_var_plot], type = "n", xlab = NA, xaxt = "n", ylim = c(0, max(max(var_true_props_avg), max_cut * 1.1)), 
				main = "Simul. Max and SE Procedures", ylab = "proportion included")
		axis(1, at = 1 : num_var_plot, labels = names(var_true_props_avg[1 : num_var_plot]), las = 2)
		
		abline(h = max_cut, col = "red")		
		for (j in 1 : num_var_plot){
			points(j, var_true_props_avg[j], pch = ifelse(var_true_props_avg[j] < max_cut, ifelse(var_true_props_avg[j] > perm_mean[j] + cover_constant * perm_se[j], 8, 1), 16))
		}		
		sapply(1 : num_var_plot, function(s){segments(s,0, x1 = s, perm_mean[s] + cover_constant * perm_se[s], col = "blue")})
		par(mar = c(5.1, 4.1, 4.1, 2.1))
		par(mfrow = c(1, 1))
	}
	
	invisible(list(
		important_vars_local_names = important_vars_pointwise_names,
		important_vars_global_max_names = important_vars_simul_max_names,
		important_vars_global_se_names = important_vars_simul_se_names,
		important_vars_local_col_nums = as.numeric(important_vars_pointwise_col_nums),
		important_vars_global_max_col_nums = as.numeric(important_vars_simul_max_col_nums),
		important_vars_global_se_col_nums = as.numeric(important_vars_simul_se_col_nums),		
		var_true_props_avg = var_true_props_avg,
		permute_mat = permute_mat
	))
}

##private
get_averaged_true_var_props = function(bart_machine, num_reps_for_avg, num_trees_for_permute){
	var_props = rep(0, bart_machine$p)
	for (i in 1 : num_reps_for_avg){
		bart_machine_dup = bart_machine_duplicate(bart_machine, num_trees = num_trees_for_permute)
		var_props = var_props + get_var_props_over_chain(bart_machine_dup)
		destroy_bart_machine(bart_machine_dup)
		cat(".")
	}
	#average over many runs
	var_props / num_reps_for_avg
}

##private
get_null_permute_var_importances = function(bart_machine, num_trees_for_permute){
	#permute the responses to disconnect x and y
	y_permuted = sample(bart_machine$y, replace = FALSE)
	
	#build BART on this permuted training data
	bart_machine_with_permuted_y = build_bart_machine(bart_machine$X, y_permuted, 
			num_trees = as.numeric(num_trees_for_permute), 
			num_burn_in = bart_machine$num_burn_in, 
			num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in,
			run_in_sample = FALSE,
			use_missing_data = bart_machine$use_missing_data,
			use_missing_data_dummies_as_covars = bart_machine$use_missing_data_dummies_as_covars,
			num_rand_samps_in_library = bart_machine$num_rand_samps_in_library,
			replace_missing_data_with_x_j_bar = bart_machine$replace_missing_data_with_x_j_bar,
			impute_missingness_with_rf_impute = bart_machine$impute_missingness_with_rf_impute,
			impute_missingness_with_x_j_bar_for_lm = bart_machine$impute_missingness_with_x_j_bar_for_lm,					
			verbose = FALSE)
	#just return the variable proportions	
	var_props = get_var_props_over_chain(bart_machine_with_permuted_y)
	destroy_bart_machine(bart_machine_with_permuted_y)
	cat(".")
	var_props
}

##private
bisectK = function(tol, coverage, permute_mat, x_left, x_right, countLimit, perm_mean, perm_se){
	count = 0
	guess = mean(c(x_left, x_right))
	while ((x_right - x_left) / 2 >= tol & count < countLimit){
		empirical_coverage = mean(sapply(1 : nrow(permute_mat), function(s){all(permute_mat[s,] - perm_mean <= guess * perm_se)}))
		if (empirical_coverage - coverage == 0){
			break
		} else if (empirical_coverage - coverage < 0){
			x_left = guess
		} else {
			x_right = guess
		}
		guess = mean(c(x_left, x_right))
		count = count + 1
	}
	guess
}


var_selection_by_permute_response_cv = function(bart_machine, k_folds = 5, num_reps_for_avg = 5, num_permute_samples = 100, num_trees_for_permute = 20, alpha = 0.05, num_trees_pred_cv = 200){
		
	if (k_folds <= 1 || k_folds > bart_machine$n){
		stop("The number of folds must be at least 2 and less than or equal to n, use \"Inf\" for leave one out")
	}	
	
	if (k_folds == Inf){ #leave-one-out
		k_folds = bart_machine$n
	}	
	
	holdout_size = round(bart_machine$n / k_folds)
	split_points = seq(from = 1, to = bart_machine$n, by = holdout_size)[1 : k_folds]
	
	L2_err_mat = matrix(NA, nrow = k_folds, ncol = 3)
	colnames(L2_err_mat) = c("important_vars_local_names", "important_vars_global_max_names", "important_vars_global _se_names")
	
	for (k in 1 : k_folds){
		cat("cv #", k, "\n", sep = "")
		#find out the indices of the holdout sample
		holdout_index_i = split_points[k]
		holdout_index_f = ifelse(k == k_folds, bart_machine$n, split_points[k + 1] - 1)
		
		#pull out training data
		training_X_k = bart_machine$model_matrix_training_data[-c(holdout_index_i : holdout_index_f), ]
		training_y_k = y[-c(holdout_index_i : holdout_index_f)]		
		
		#make a temporary bart machine just so we can run the var selection for all three methods
		bart_machine_temp = build_bart_machine(as.data.frame(training_X_k), training_y_k, 				 
				num_trees = bart_machine$num_trees, 
				num_burn_in = bart_machine$num_burn_in,
				cov_prior_vec = bart_machine$cov_prior_vec,
				run_in_sample = FALSE,
				use_missing_data = bart_machine$use_missing_data,
				use_missing_data_dummies_as_covars = bart_machine$use_missing_data_dummies_as_covars,
				num_rand_samps_in_library = bart_machine$num_rand_samps_in_library,
				replace_missing_data_with_x_j_bar = bart_machine$replace_missing_data_with_x_j_bar,
				impute_missingness_with_rf_impute = bart_machine$impute_missingness_with_rf_impute,
				impute_missingness_with_x_j_bar_for_lm = bart_machine$impute_missingness_with_x_j_bar_for_lm,	
				verbose = FALSE)
		bart_variables_select_obj_k = var_selection_by_permute_response_three_methods(bart_machine_temp, 
				num_permute_samples = num_permute_samples, 
				num_trees_for_permute = num_trees_for_permute, 
				alpha = alpha, 
				plot = FALSE)		
		destroy_bart_machine(bart_machine_temp)
		
		#pull out test data
		test_X_k = bart_machine$model_matrix_training_data[holdout_index_i : holdout_index_f, ]
		text_y_k = y[holdout_index_i : holdout_index_f]
		
		cat("method")
		for (method in colnames(L2_err_mat)){
			cat(".")
			#pull out the appropriate vars
			vars_selected_by_method = bart_variables_select_obj_k[[method]]
			if (length(vars_selected_by_method) == 0){
				#we just predict ybar
				ybar_est = mean(training_y_k)
				#and we take L2 error against ybar
				L2_err_mat[k, method] = sum((text_y_k - ybar_est)^2)
			} else {
				training_X_k_red_by_vars_picked_by_method = as.data.frame(training_X_k[, vars_selected_by_method])
				#now build the bart machine based on reduced model
				bart_machine_temp = build_bart_machine(training_X_k_red_by_vars_picked_by_method, training_y_k,
						num_burn_in = bart_machine$num_burn_in,
						num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in,
						num_trees = num_trees_pred_cv,
						run_in_sample = FALSE,
						verbose = FALSE)
				#and calculate oos-L2 and cleanup
				test_X_k_red_by_vars_picked_by_method = as.data.frame(test_X_k[, bart_variables_select_obj_k[[method]]])
				predict_obj = bart_predict_for_test_data(bart_machine_temp, test_X_k_red_by_vars_picked_by_method, text_y_k)
				destroy_bart_machine(bart_machine_temp)
				#now record it
				L2_err_mat[k, method] = predict_obj$L2_err
			}
		}
		cat("\n")
	}
	
	#now extract the lowest oos-L2 to find the "best" method for variable selection
	L2_err_by_method = colSums(L2_err_mat)
	min_var_selection_method = colnames(L2_err_mat)[which(L2_err_by_method == min(L2_err_by_method))]
	min_var_selection_method = min_var_selection_method[1]

	#now (finally) do var selection on the entire data and then return the vars from the best method found via cross-validation
	cat("final", "\n")
	bart_variables_select_obj = var_selection_by_permute_response_three_methods(bart_machine, 
			num_permute_samples = num_permute_samples, 
			num_trees_for_permute = num_trees_for_permute, 
			alpha = alpha, 
			plot = FALSE)
	
	list(best_method = min_var_selection_method, important_vars_cv = sort(bart_variables_select_obj[[min_var_selection_method]]))
}