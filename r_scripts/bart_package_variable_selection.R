
#methods: pointwise
var_selection_by_permute_response = function(bart_machine, num_reps_for_avg = 5, num_permute_samples = 100, num_trees_for_permute = 20, alpha = 0.05, plot = TRUE, num_var_plot = Inf){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	
	permute_mat = matrix(NA, nrow = num_permute_samples, ncol = bart_machine$p)
	colnames(permute_mat) = bart_machine$training_data_features
	
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
	important_vars_pointwise = names(var_true_props_avg[var_true_props_avg > pointwise_cutoffs])		

	max_cut = quantile(apply(permute_mat, 1 ,max), 1 - alpha)
	important_vars_simul_max = names(var_true_props_avg[var_true_props_avg >= max_cut])		

	perm_se = apply(permute_mat, 2, sd)
	perm_mean = apply(permute_mat, 2, mean)
	cover_constant = bisectK(tol = .01 , coverage = 1 - alpha, permute_mat = permute_mat, x_left = 1, x_right = 20, countLimit = 100, perm_mean = perm_mean, perm_se = perm_se)
	important_vars_simul_se = names(var_true_props_avg[which(var_true_props_avg >= perm_mean + cover_constant * perm_se)])	
	
	

	if (plot){
		#sort attributes by most important
		
		
		if (num_var_plot == Inf | num_var_plot > bart_machine$p){
			num_var_plot = bart_machine$p
		}
		
		par(mfrow = c(2, 1))
		##pointwise plot
		plot(1 : num_var_plot, var_true_props_avg[1 : num_var_plot], type = "n", xlab = NA, xaxt = "n", ylim = c(0, max(max(var_true_props_avg), max_cut * 1.1)),
				main = "Variable Selection by Pointwise Method", ylab = "proportion included")
		axis(1, at = 1 : num_var_plot, labels = names(var_true_props_avg[1 : num_var_plot]), las = 2)
		for (j in 1 : num_var_plot){
			points(j, var_true_props_avg[j], pch = ifelse(var_true_props_avg[j] < quantile(permute_mat[, j], 1 - alpha), 1, 16))
		}
		
		sapply(1 : num_var_plot, function(s){segments(s, 0, x1 = s, quantile(permute_mat[, s], 1 - alpha), col = "forestgreen")})
		
		##simul plots
		plot(1 : num_var_plot, var_true_props_avg[1 : num_var_plot], type = "n", xlab = NA, xaxt = "n", ylim = c(0, max(max(var_true_props_avg), max_cut * 1.1)), 
				main = "Variable Selection by Simultaneous Max and SE Methods", ylab = "proportion included")
		axis(1, at = 1 : num_var_plot, labels = names(var_true_props_avg[1 : num_var_plot]), las = 2)
		
		abline(h = max_cut, col = "red")		
		for (j in 1 : num_var_plot){
			points(j, var_true_props_avg[j], pch = ifelse(var_true_props_avg[j] < max_cut, ifelse(var_true_props_avg[j] > perm_mean[j] + cover_constant * perm_se[j], 8, 1), 16))
		}		
		sapply(1 : num_var_plot, function(s){segments(s,0, x1 = s, perm_mean[s] + cover_constant * perm_se[s], col = "blue")})
		
	}
	
	invisible(list(
		important_vars_pointwise = important_vars_pointwise,
		important_vars_simul_max = important_vars_simul_max,
		important_vars_simul_se = important_vars_simul_se,
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
			verbose = FALSE)
#	#just return the variable proportions	
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



var_importance_by_dropping_variable = function(bart_machine, list_of_vars = NULL, holdout_pctg = 0.2, plot = TRUE, num_var_plot = Inf){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	
	if (is.null(list_of_vars)){
		list_of_vars = 1 : bart_machine$p
	}
	prev_cov_prior_vec = bart_machine$cov_prior_vec
	if (is.null(prev_cov_prior_vec)){
		prev_cov_prior_vec = rep(1, bart_machine$p)
	}
	out_of_sample_indices = sort(sample(1 : bart_machine$n, round(bart_machine$n * holdout_pctg)))
	training_data_X = bart_machine$X[setdiff(1 : bart_machine$n, out_of_sample_indices), ]
	training_data_y = bart_machine$y[setdiff(1 : bart_machine$n, out_of_sample_indices)]
	test_data_X = bart_machine$X[out_of_sample_indices, ]
	test_data_y = bart_machine$y[out_of_sample_indices]
	
	rmse_pct_change = array(NA, length(list_of_vars))
	names(rmse_pct_change) = bart_machine$training_data_features[list_of_vars]
	
	#first run one bart out of sample
	bart_machine_dup = bart_machine_duplicate(bart_machine, X = training_data_X, y = training_data_y)
	predict_obj = bart_predict_for_test_data(bart_machine_dup, test_data_X, test_data_y)
	full_rmse = predict_obj$rmse
	destroy_bart_machine(bart_machine_dup)
	
	for (j in 1 : length(list_of_vars)){
		var = list_of_vars[j]
		cov_prior_vec = prev_cov_prior_vec
		cov_prior_vec[var] = 1 / 1000000000000 #effectively zero
		bart_machine_dup = bart_machine_duplicate(bart_machine, X = training_data_X, y = training_data_y, cov_prior_vec = cov_prior_vec)
		predict_obj = bart_predict_for_test_data(bart_machine_dup, test_data_X, test_data_y)		
		rmse_pct_change[j] = (predict_obj$rmse - full_rmse) / full_rmse * 100
		destroy_bart_machine(bart_machine_dup)
		cat(".")
		
	}
	cat("\n")
	
	if (plot){
		if (num_var_plot == Inf){
			num_var_plot = length(list_of_vars)
		}	
		rmse_pct_change_to_plot = sort(rmse_pct_change, decr = T)[1 : num_var_plot]
		barplot(rmse_pct_change_to_plot, ylab = "RMSE Degradation (%)", xlab = "Variable", main = "Variable Importance by Dropping Variable", las = 2)
	}
	
	invisible(rmse_pct_change)
}

var_importance_by_shuffling = function(bart_machine, list_of_vars = NULL, holdout_pctg = 0.2, plot = TRUE, num_var_plot = Inf){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	
	if (is.null(list_of_vars)){
		list_of_vars = 1 : bart_machine$p
	}
	
	
	
	out_of_sample_indices = sort(sample(1 : bart_machine$n, round(bart_machine$n * holdout_pctg)))
	training_data_X = bart_machine$X[setdiff(1 : bart_machine$n, out_of_sample_indices), ]
	training_data_y = bart_machine$y[setdiff(1 : bart_machine$n, out_of_sample_indices)]
	test_data_X = bart_machine$X[out_of_sample_indices, ]
	test_data_y = bart_machine$y[out_of_sample_indices]
	
	rmse_pct_change = array(NA, length(list_of_vars))
	names(rmse_pct_change) = bart_machine$training_data_features[list_of_vars]
	
	#first run one bart out of sample
	bart_machine_dup = bart_machine_duplicate(bart_machine, X = training_data_X, y = training_data_y)
	predict_obj = bart_predict_for_test_data(bart_machine_dup, test_data_X, test_data_y)
	full_rmse = predict_obj$rmse
	
	
	for (j in 1 : length(list_of_vars)){
		var = list_of_vars[j]
		#shuffle that var's column
		test_data_X_shuffled = test_data_X
		test_data_X_shuffled[, var] = sample(test_data_X_shuffled[, var], replace = FALSE)
		
		predict_obj = bart_predict_for_test_data(bart_machine_dup, test_data_X_shuffled, test_data_y)		
		rmse_pct_change[j] = (predict_obj$rmse - full_rmse) / full_rmse * 100
		cat(".")
		
	}
	cat("\n")
	destroy_bart_machine(bart_machine_dup)
	
	if (plot){
		if (num_var_plot == Inf){
			num_var_plot = length(list_of_vars)
		}
		
		rmse_pct_change_to_plot = sort(rmse_pct_change, decr = T)[1 : num_var_plot]
		barplot(rmse_pct_change_to_plot, ylab = "RMSE Degradation (%)", xlab = "Variable", 
				main = "Variable Importance by Shuffling Out-of-Sample", las = 2)
	}
	
	invisible(rmse_pct_change)
}