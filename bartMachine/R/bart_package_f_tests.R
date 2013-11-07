
cov_importance_test = function(bart_machine, covariates = NULL, num_permutations = 100, num_trees = NULL, plot = TRUE){
	if (is.null(covariates)){
		title = "BART omnibus test for covariate importance\n"
	} else if (length(covariates) <= 3){
		if (class(covariates[1]) == "numeric"){
			cov_names = paste(bart_machine$training_data_features_with_missing_features[covariates], collapse = ", ")
		} else {
			cov_names = paste(covariates, collapse = ", ")
		}
		title = paste("BART test for importance of covariate(s):", cov_names, "\n")
	} else {
		title = paste("BART test for importance of", length(covariates), "covariates", "\n")
	}
	cat(title)
	sd_y = sd(bart_machine$y)
	
	if (is.null(num_trees)){
		num_trees = bart_machine$num_trees
		observed_error_estimate = ifelse(bart_machine$pred_type == "regression", bart_machine$PseudoRsq, bart_machine$misclassification_error)
	} else {
		bart_machine_copy = build_bart_machine(X = bart_machine$X, y = bart_machine$y, 
				use_missing_data = bart_machine$use_missing_data,
				use_missing_data_dummies_as_covars = bart_machine$use_missing_data_dummies_as_covars, 
				num_trees = num_trees, 
				verbose = FALSE) #we have to turn verbose off otherwise there would be too many outputs
		observed_error_estimate = ifelse(bart_machine$pred_type == "regression", bart_machine$PseudoRsq, bart_machine$misclassification_error)
		destroy_bart_machine(bart_machine_copy)	
	}
	
	permutation_samples_of_error = array(NA, num_permutations)
	for (nsim in 1 : num_permutations){
		cat(".")
		if (nsim %% 50 == 0){
			cat("\n")
		}	
		#omnibus F-like test - just permute y (same as permuting ALL the columns of X and it's faster)
		if (is.null(covariates)){
			bart_machine_samp = build_bart_machine(X = bart_machine$X, y = sample(bart_machine$y), 
					use_missing_data = bart_machine$use_missing_data,
					use_missing_data_dummies_as_covars = bart_machine$use_missing_data_dummies_as_covars,
					num_trees = num_trees, 
					verbose = FALSE) #we have to turn verbose off otherwise there would be too many outputs
		#partial F-like test - permute the columns that we're interested in seeing if they matter
		} else {
			X_samp = bart_machine$X #copy original design matrix

			bart_machine_samp = build_bart_machine(X = X_samp, y = bart_machine$y, 
					covariates_to_permute = covariates,
					use_missing_data = bart_machine$use_missing_data,
					use_missing_data_dummies_as_covars = bart_machine$use_missing_data_dummies_as_covars,
					num_trees = num_trees, 
					verbose = FALSE) #we have to turn verbose off otherwise there would be too many outputs
		}
		#record permutation result
		permutation_samples_of_error[nsim] = ifelse(bart_machine$pred_type == "regression", bart_machine_samp$PseudoRsq, bart_machine_samp$misclassification_error)
		destroy_bart_machine(bart_machine_samp)		
	}
	cat("\n")
	
	pval = ifelse(bart_machine$pred_type == "regression", sum(observed_error_estimate < permutation_samples_of_error), sum(observed_error_estimate > permutation_samples_of_error)) / num_permutations
	
	if (plot){
		hist(permutation_samples_of_error, 
				xlim = c(min(permutation_samples_of_error, 0.99 * observed_error_estimate), max(permutation_samples_of_error, 1.01 * observed_error_estimate)),
				xlab = paste("permutation samples\n pval = ", pval),
				br = num_permutations / 10,
				main = paste(title, "Null Samples of", ifelse(bart_machine$pred_type == "regression", "Pseudo-R^2's", "Misclassification Errors")))
		abline(v = observed_error_estimate, col = "blue", lwd = 3)
	}
	cat("p_val = ", pval, "\n")
	invisible(list(scaled_rmse_perm_samples = permutation_samples_of_error, scaled_rmse_obs = observed_error_estimate, pval = pval))
}

