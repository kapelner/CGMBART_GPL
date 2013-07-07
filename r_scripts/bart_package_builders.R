DEFAULT_ALPHA = 0.95
DEFAULT_BETA = 2
DEFAULT_K = 2
DEFAULT_Q = 0.9
DEFAULT_NU = 3.0
DEFAULT_PROB_STEPS = c(2.5, 2.5, 4) / 9
DEFAULT_PROB_RULE_CLASS = 0.5

build_bart_machine = function(X = NULL, y = NULL, Xy = NULL, 
		num_trees = 200, 
		num_burn_in = 250, 
		num_iterations_after_burn_in = 1000, 
		alpha = DEFAULT_ALPHA,
		beta = DEFAULT_BETA,
		k = DEFAULT_K,
		q = DEFAULT_Q,
		nu = DEFAULT_NU,
		prob_rule_class = DEFAULT_PROB_RULE_CLASS,
		mh_prob_steps = DEFAULT_PROB_STEPS,
		debug_log = FALSE,
		run_in_sample = TRUE,
		s_sq_y = "mse", # "mse" or "var"
		unique_name = "unnamed",
		print_tree_illustrations = FALSE,
		cov_prior_vec = NULL,
		use_missing_data = FALSE,
		use_missing_data_dummies_as_covars = FALSE,
		replace_missing_data_with_x_j_bar = FALSE,
		impute_missingness_with_rf_impute = FALSE,
		impute_missingness_with_x_j_bar_for_lm = TRUE,
		mem_cache_for_speed = TRUE,
		verbose = TRUE){
	
	t0 = Sys.time()
	#immediately initialize Java
	init_java_for_bart()
	
	if ((is.null(X) && is.null(Xy)) || is.null(y) && is.null(Xy)){
		stop("You need to give BART a training set either by specifying X and y or by specifying a matrix Xy which contains the response named \"y.\"\n")
	} else if (is.null(X) && is.null(y)){ #they specified Xy, so now just pull out X,y
		y = Xy[, ncol(Xy)]
		for (j in 1 : (ncol(Xy) - 1)){
			if (colnames(Xy)[j] == ""){
				colnames(Xy)[j] = paste("V", j, sep = "")
			}
		}
		X = as.data.frame(Xy[, 1 : (ncol(Xy) - 1)])
		colnames(X) = colnames(Xy)[1 : (ncol(Xy) - 1)]
	}
	
	#now take care of classification or regression
	y_levels = levels(y)
	if (class(y) == "numeric"){ #if y is numeric, then it's a regression problem
		java_bart_machine = .jnew("CGM_BART.CGMBARTRegressionMultThread")
		y_numeric = y
		pred_type = "regression"
	} else if (class(y) == "factor" & length(y_levels) == 2){ #if y is a factor and binary
		java_bart_machine = .jnew("CGM_BART.CGMBARTClassificationMultThread")
		y_numeric = ifelse(y == y_levels[1], 0, 1)
		pred_type = "classification"
	} else if (class(y) == "factor" & length(y_levels) == 3){
		stop("Please use the function \"--\" for trinomial classification.\n")
	} else if (class(y) == "integer"){
		stop("Please use the function \"--\" for ordinal classification.\n")
	} else { #otherwise throw an error
		stop("Your response must be either numeric, a factor with two or three levels or an integer.\n")
	}	
	
	
	num_gibbs = num_burn_in + num_iterations_after_burn_in
	
	#R loves to convert 1-column matrices into vectors, so just convert it on back
	if (class(X) == "numeric"){
		X = as.data.frame(as.matrix(X))
	}
	
	if (ncol(X) == 0){
		stop("Your data matrix must have at least one attribute.")
	}
	if (nrow(X) == 0){
		stop("Your data matrix must have at least one observation.")
	}
	if (length(y) != nrow(X)){
		stop("The number of responses must be equal to the number of observations in the training data.")
	}
	
	#if no column names, make up names
	if (is.null(colnames(X))){
		colnames(X) = paste("V", seq(from = 1, to = ncol(X), by = 1), sep = "")
	}
	
	#check for errors in data
	if (check_for_errors_in_training_data(X)){
		return;
	}
	
	if (length(na.omit(y_numeric)) != length(y_numeric)){
		stop("You cannot have any missing data in your response vector.")
	}
	
	rf_imputations_for_missing = NULL
	if (impute_missingness_with_rf_impute){
		if (nrow(na.omit(X)) == nrow(X)){ #for the cases where it doesn't impute
			warning("No missing entries in the training data to impute.")
			rf_imputations_for_missing = X
		} else {
			rf_imputations_for_missing = rfImpute(X, y)
			rf_imputations_for_missing = rf_imputations_for_missing[, 2 : ncol(rf_imputations_for_missing)]	
		}
		colnames(rf_imputations_for_missing) = paste(colnames(rf_imputations_for_missing), "_imp", sep = "")
	}

	model_matrix_training_data = cbind(pre_process_training_data(X, use_missing_data_dummies_as_covars, rf_imputations_for_missing, verbose), y_numeric)
	
	#if we're not using missing data, go on and nuke it
	if (!use_missing_data && !replace_missing_data_with_x_j_bar){
		rows_before = nrow(model_matrix_training_data)
		data = na.omit(model_matrix_training_data)
		rows_after = nrow(model_matrix_training_data)
		if (verbose && rows_before - rows_after > 0){
			cat("Deleted", rows_before - rows_after, "row(s) due to missing data. Try turning missing data feature on next time. ")
		}
	} else if (replace_missing_data_with_x_j_bar){
		model_matrix_training_data = imputeMatrixByXbarj(model_matrix_training_data, model_matrix_training_data)
		if (verbose){
			cat("Imputed missing data using attribute averages. ")
		}
	}
	
	
	#first set the name
	.jcall(java_bart_machine, "V", "setUniqueName", unique_name)
	#now set whether we want the program to log to a file
	if (debug_log & verbose){
		cat("warning: printing out the log file will slow down the runtime significantly\n")
		.jcall(java_bart_machine, "V", "writeStdOutToLogFile")
	}
	#set whether we want there to be tree illustrations
	if (print_tree_illustrations & verbose){
		cat("warning: printing tree illustrations will slow down the runtime significantly\n")
		.jcall(java_bart_machine, "V", "printTreeIllustations")
	}
	
	#set the std deviation of y to use
	if (ncol(model_matrix_training_data) - 1 >= nrow(model_matrix_training_data)){
		if (verbose){
			cat("warning: cannot use MSE of linear model for s_sq_y if p > n\n")
		}
		s_sq_y = "var"
		
	}
	
	sig_sq_est = NULL
	if (pred_type == "regression"){
		y_range = max(y) - min(y)
		y_trans = (y - min(y)) / y_range - 0.5
		if (s_sq_y == "mse"){
			X_for_lm = as.data.frame(model_matrix_training_data)[1 : (ncol(model_matrix_training_data) - 1)]
			if (impute_missingness_with_x_j_bar_for_lm){
				X_for_lm = imputeMatrixByXbarj(X_for_lm, X_for_lm)
			}
			mod = lm(y_trans ~ ., X_for_lm)
			mse = var(mod$residuals)
			sig_sq_est = as.numeric(mse)
			.jcall(java_bart_machine, "V", "setSampleVarY", sig_sq_est)
		} else if (s_sq_y == "var"){
			sig_sq_est = as.numeric(var(y_trans))
			.jcall(java_bart_machine, "V", "setSampleVarY", sig_sq_est)
		} else {
			stop("s_sq_y must be \"rmse\" or \"sd\"", call. = FALSE)
			return(TRUE)
		}
		sig_sq_est = sig_sq_est * y_range^2		
	}
	
	
	#build bart to spec with what the user wants
	.jcall(java_bart_machine, "V", "setNumCores", as.integer(BART_NUM_CORES)) #this must be set FIRST!!!
	.jcall(java_bart_machine, "V", "setNumTrees", as.integer(num_trees))
	.jcall(java_bart_machine, "V", "setNumGibbsBurnIn", as.integer(num_burn_in))
	.jcall(java_bart_machine, "V", "setNumGibbsTotalIterations", as.integer(num_gibbs))
	.jcall(java_bart_machine, "V", "setAlpha", alpha)
	.jcall(java_bart_machine, "V", "setBeta", beta)
	.jcall(java_bart_machine, "V", "setK", k)
	.jcall(java_bart_machine, "V", "setQ", q)
	.jcall(java_bart_machine, "V", "setNU", nu)
	mh_prob_steps = mh_prob_steps / sum(mh_prob_steps) #make sure it's a prob vec
	.jcall(java_bart_machine, "V", "setProbGrow", mh_prob_steps[1])
	.jcall(java_bart_machine, "V", "setProbPrune", mh_prob_steps[2])
	.jcall(java_bart_machine, "V", "setVerbose", verbose)
	.jcall(java_bart_machine, "V", "setMemCacheForSpeed", mem_cache_for_speed)
	
	
	if (length(cov_prior_vec) != 0){
		#put in checks here for user to make sure the covariate prior vec is the correct length
		offset = length(cov_prior_vec) - (ncol(model_matrix_training_data) - 1) 
		if (offset < 0){
			warning(paste("covariate prior vector length =", length(cov_prior_vec), "has to be equal to p =", ncol(model_matrix_training_data) - 1, "the vector was lengthened (with 1's)"))
			cov_prior_vec = c(cov_prior_vec, rep(1, -offset))
		}
		if (length(cov_prior_vec) != ncol(model_matrix_training_data) - 1){
			warning(paste("covariate prior vector length =", length(cov_prior_vec), "has to be equal to p =", ncol(model_matrix_training_data) - 1, "the vector was shortened"))
			cov_prior_vec = cov_prior_vec[1 : (ncol(model_matrix_training_data) - 1)]		
		}		
		if (sum(cov_prior_vec > 0) != ncol(model_matrix_training_data) - 1){
			stop("covariate prior vector has to have all its elements be positive", call. = FALSE)
			return(TRUE)
		}
		.jcall(java_bart_machine, "V", "setCovSplitPrior", as.numeric(cov_prior_vec))
	}
	
	#now load the training data into BART
	for (i in 1 : nrow(model_matrix_training_data)){
		.jcall(java_bart_machine, "V", "addTrainingDataRow", as.character(model_matrix_training_data[i, ]))
	}
	.jcall(java_bart_machine, "V", "finalizeTrainingData")
	
	#build the bart machine and let the user know what type of BART this is
	if (verbose){
		cat("Building BART for", pred_type, "...")
		if (length(cov_prior_vec) != 0){
			cat("Covariate importance prior ON. ")
		}
		if (use_missing_data){
			cat("Missing data feature ON. ")
		}
		if (use_missing_data_dummies_as_covars){
			cat("Missingness used as covariates. ")
		}
		if (impute_missingness_with_rf_impute){
			cat("Missing values imputed via rfImpute. ")
		}
		cat("\n")
	}
	.jcall(java_bart_machine, "V", "Build")
	
	#now once it's done, let's extract things that are related to diagnosing the build of the BART model
	p = ncol(model_matrix_training_data) - 1 # we subtract one because we tacked on the response as the last column
	bart_machine = list(java_bart_machine = java_bart_machine,
			training_data_features = colnames(model_matrix_training_data)[1 : ifelse(use_missing_data, (p / 2), p)],
			training_data_features_with_missing_features = colnames(model_matrix_training_data)[1 : p],
			X = X,
			y = y,
			y_levels = y_levels,
			pred_type = pred_type,
			model_matrix_training_data = model_matrix_training_data,
			n = nrow(model_matrix_training_data),
			p = p,
			num_cores = BART_NUM_CORES,
			num_trees = num_trees,
			num_burn_in = num_burn_in,
			num_iterations_after_burn_in = num_iterations_after_burn_in, 
			num_gibbs = num_gibbs,
			alpha = alpha,
			beta = beta,
			k = k,
			q = q,
			nu = nu,
			prob_rule_class = prob_rule_class,
			mh_prob_steps = mh_prob_steps,
			s_sq_y = s_sq_y,
			run_in_sample = run_in_sample,
			cov_prior_vec = cov_prior_vec,
			sig_sq_est = sig_sq_est,
			time_to_build = Sys.time() - t0,
			use_missing_data = use_missing_data,
			replace_missing_data_with_x_j_bar = replace_missing_data_with_x_j_bar,
			add_imputations = impute_missingness_with_rf_impute,
			verbose = verbose,
			bart_destroyed = FALSE
	)
	
	#once its done gibbs sampling, see how the training data does if user wants
	if (run_in_sample){
		if (verbose){
			cat("evaluating in sample data...")
		}
		if (pred_type == "regression"){
			y_hat_posterior_samples = 
					t(sapply(.jcall(bart_machine$java_bart_machine, "[[D", "getGibbsSamplesForPrediction", .jarray(model_matrix_training_data, dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))
			
			#to get y_hat.. just take straight mean of posterior samples
			y_hat_train = rowMeans(y_hat_posterior_samples)
			#return a bunch more stuff
			bart_machine$y_hat_train = y_hat_train
			bart_machine$residuals = y - bart_machine$y_hat_train
			bart_machine$L1_err_train = sum(abs(bart_machine$residuals))
			bart_machine$L2_err_train = sum(bart_machine$residuals^2)
			bart_machine$PseudoRsq = 1 - bart_machine$L2_err_train / sum((y - mean(y))^2) #pseudo R^2 acc'd to our dicussion with Ed and Shane
			bart_machine$rmse_train = sqrt(bart_machine$L2_err_train / bart_machine$n)
		} else if (pred_type == "classification"){
			p_hat_posterior_samples = 
					t(sapply(.jcall(bart_machine$java_bart_machine, "[[D", "getGibbsSamplesForPrediction", .jarray(model_matrix_training_data, dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))
			
			#to get y_hat.. just take straight mean of posterior samples
			p_hat_train = rowMeans(p_hat_posterior_samples)
			y_hat_train = ifelse(p_hat_train > prob_rule_class, y_levels[2], y_levels[1])
			#return a bunch more stuff
			bart_machine$p_hat_train = p_hat_train
			bart_machine$y_hat_train = y_hat_train
			
			#calculate confusion matrix
			confusion_matrix = as.data.frame(matrix(NA, nrow = 3, ncol = 3))
			rownames(confusion_matrix) = c(paste("actual", y_levels), "use errors")
			colnames(confusion_matrix) = c(paste("predicted", y_levels), "model errors")
			
			confusion_matrix[1 : 2, 1 : 2] = as.integer(table(y, y_hat_train)) 
			confusion_matrix[3, 1] = round(confusion_matrix[2, 1] / (confusion_matrix[1, 1] + confusion_matrix[2, 1]), 3)
			confusion_matrix[3, 2] = round(confusion_matrix[1, 2] / (confusion_matrix[1, 2] + confusion_matrix[2, 2]), 3)
			confusion_matrix[1, 3] = round(confusion_matrix[1, 2] / (confusion_matrix[1, 1] + confusion_matrix[1, 2]), 3)
			confusion_matrix[2, 3] = round(confusion_matrix[2, 1] / (confusion_matrix[2, 1] + confusion_matrix[2, 2]), 3)
			confusion_matrix[3, 3] = round((confusion_matrix[1, 2] + confusion_matrix[2, 1]) / sum(confusion_matrix[1 : 2, 1 : 2]), 3)
			
			bart_machine$confusion_matrix = confusion_matrix
			bart_machine$misclassification_error = confusion_matrix[3, 3]
		}
		if (verbose){
			cat("done\n")
		}
	}
	
	#use R's S3 object orientation
	class(bart_machine) = "bart_machine"
	bart_machine
}

build_ordinal_bart_machine = function(X = NULL, y = NULL, Xy = NULL, 
		num_trees = 200, 
		num_burn_in = 250, 
		num_iterations_after_burn_in = 1000, 
		alpha = DEFAULT_ALPHA,
		beta = DEFAULT_BETA,
		k = DEFAULT_K,
		q = DEFAULT_Q,
		nu = DEFAULT_NU,
		mh_prob_steps = DEFAULT_PROB_STEPS,
		debug_log = FALSE,
		run_in_sample = TRUE,
		unique_name = "unnamed",
		print_tree_illustrations = FALSE,
		cov_prior_vec = NULL,
		use_missing_data = TRUE,
		mem_cache_for_speed = TRUE,
		verbose = TRUE){
	
	if (class(y) != "integer"){
		stop("Please convert the response to an integer for ordinal classification.\n")
	}
	
	t0 = Sys.time()
	#immediately initialize Java
	init_java_for_bart()
	
	if ((is.null(X) && is.null(Xy)) || is.null(y) && is.null(Xy)){
		stop("You need to give BART a training set either by specifying X and y or by specifying a matrix Xy which contains the response named \"y.\"\n")
	} else if (is.null(X) && is.null(y)){ #they specified Xy, so now just pull out X,y
		y = Xy[, ncol(Xy)]
		for (j in 1 : (ncol(Xy) - 1)){
			if (colnames(Xy)[j] == ""){
				colnames(Xy)[j] = paste("V", j, sep = "")
			}
		}
		X = as.data.frame(Xy[, 1 : (ncol(Xy) - 1)])
		colnames(X) = colnames(Xy)[1 : (ncol(Xy) - 1)]
	}	
	
	y_levels = as.integer(names(table(y)))
	print(y_levels)
	
	#checks for rare observations
	if (min(table(y)) <= 5){
		warning("Rare outcome class exists")
	}
	
	bart_machines = list()
	for (kc in 1 : (length(y_levels) - 1)){
		print(kc)
		cat(y_levels[kc], "vs. those higher: ")
		
		#now we have to "make up 0/1" binary classifications
		y0 = y[y == y_levels[kc]]
		X0 = X[y == y_levels[kc], ]
		y1 = y[y > y_levels[kc]]
		X1 = X[y > y_levels[kc], ]
		
		Xsub = rbind(X0, X1)
		ysub = as.factor(c(rep(0, length(y0)), rep(1, length(y1))))
		
		
		bart_machines[[kc]] = build_bart_machine(X = Xsub, y = ysub, 
			num_trees = num_trees, 
			num_burn_in = num_burn_in, 
			num_iterations_after_burn_in = num_iterations_after_burn_in, 
			alpha = alpha,
			beta = beta,
			k = k,
			q = q,
			nu = nu,
			mh_prob_steps = mh_prob_steps,
			debug_log = debug_log,
			run_in_sample = TRUE,
			unique_name = unique_name,
			print_tree_illustrations = print_tree_illustrations,
			cov_prior_vec = cov_prior_vec,
			use_missing_data = use_missing_data,
			mem_cache_for_speed = mem_cache_for_speed,
			verbose = verbose)
	
		
	}
	
	
	bart_machines	
}

bart_machine_duplicate = function(bart_machine, X = NULL, y = NULL, cov_prior_vec = NULL, num_trees = NULL, run_in_sample = NULL, ...){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	if (is.null(X)){
		X = bart_machine$X
	}
	if (is.null(y)){
		y = bart_machine$y
	}
	if (is.null(cov_prior_vec)){
		cov_prior_vec = bart_machine$cov_prior_vec
	}
	if (is.null(num_trees)){
		num_trees = bart_machine$num_trees
	}	
	if (is.null(run_in_sample)){
		run_in_sample = FALSE
	}
	build_bart_machine(X, y,
			num_trees = num_trees,
			num_burn_in = bart_machine$num_burn_in, 
			num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in, 
			alpha = bart_machine$alpha,
			beta = bart_machine$beta,
			debug_log = FALSE,
			s_sq_y = bart_machine$s_sq_y,
			#num_cores = bart_machine$num_cores,
			cov_prior_vec = cov_prior_vec,
			print_tree_illustrations = FALSE,
			run_in_sample = run_in_sample,
			verbose = FALSE, 
			...)
}

destroy_bart_machine = function(bart_machine){
	.jcall(bart_machine$java_bart_machine, "V", "destroy")
	bart_machine$bart_destroyed = TRUE
	#explicitly ask the JVM to give use the RAM back right now
	.jcall("java/lang/System", "V", "gc")
}

imputeMatrixByXbarj = function(X_with_missing, X_for_calculating_avgs){
	for (i in 1 : nrow(X_with_missing)){
		for (j in 1 : ncol(X_with_missing)){
			if (is.na(X_with_missing[i, j])){
				X_with_missing[i, j] = mean(X_for_calculating_avgs[, j], na.rm = TRUE)
			}
		}
	}
	X_with_missing
}