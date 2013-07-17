
cov_importance_test = function(bart_machine, covariates = NULL, num_permutations = 200, num_trees = NULL, plot = TRUE){
	if (is.null(covariates)){
		title = "BART omnibus test for covariate importance\n"
	} else {
		title = paste("BART test for importance of covariate(s):", paste(bart_machine$training_data_features[covariates], collapse = ", "), "\n")
	}
	cat(title)
	sd_y = sd(bart_machine$y)
	
	if (is.null(num_trees)){
		num_trees = bart_machine$num_trees
		pseudoRsq_obs = bart_machine$PseudoRsq
	} else {
		bart_machine_copy = build_bart_machine(X = bart_machine$X, y = bart_machine$y, num_trees = num_trees, verbose = FALSE)
		pseudoRsq_obs = bart_machine_copy$PseudoRsq
		destroy_bart_machine(bart_machine_copy)	
	}
	
	pseudoRsq_perm_samples = array(NA, num_permutations)
	for (nsim in 1 : num_permutations){
		cat(".")
		if (nsim %% 50 == 0){
			cat("\n")
		}	
		#omnibus F-like test - just permute y (same as permuting ALL the columns of X and it's faster)
		if (is.null(covariates)){
			bart_machine_samp = build_bart_machine(X = bart_machine$X, y = sample(bart_machine$y), num_trees = num_trees, verbose = FALSE)
		#partial F-like test - permute the columns that we're interested in seeing if they matter
		} else {
			X_samp = bart_machine$X #copy original design matrix
			for (j in covariates){
				X_samp[, j] = sample(X_samp[, j])
			}
			bart_machine_samp = build_bart_machine(X = X_samp, y = bart_machine$y, num_trees = num_trees, verbose = FALSE)
		}
		#record permutation result
		pseudoRsq_perm_samples[nsim] = bart_machine_samp$PseudoRsq
		destroy_bart_machine(bart_machine_samp)		
	}
	cat("\n")
	
	pval = sum(pseudoRsq_obs < pseudoRsq_perm_samples) / num_permutations
	
	if (plot){
		hist(pseudoRsq_perm_samples, 
				xlim = c(min(pseudoRsq_perm_samples, 0.99 * pseudoRsq_obs), max(pseudoRsq_perm_samples, 1.01 * pseudoRsq_obs)),
				xlab = paste("permutation samples\n pval = ", pval),
				br = num_permutations / 5,
				main = paste(title, "Null Samples of Pseudo-R^2's"))
		abline(v = pseudoRsq_obs, col = "blue")
	}
	cat("p_val = ", pval, "\n")
	invisible(list(scaled_rmse_perm_samples = pseudoRsq_perm_samples, scaled_rmse_obs = pseudoRsq_obs, pval = pval))
}

