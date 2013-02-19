#libraries and dependencies
tryCatch(library(rJava), error = function(e){install.packages("rJava")}, finally = library(rJava))

#
#if (.Platform$OS.type == "windows"){
#	tryCatch(library(rJava), error = function(e){install.packages("rJava")}, finally = library(rJava))
#	tryCatch(library(BayesTree), error = function(e){install.packages("BayesTree")}, finally = library(BayesTree))
#} else {
#	tryCatch(library(rJava), error = function(e){library(rJava, lib.loc = "~/R/")})	
#	library(BayesTree, lib.loc = "~/R/")
#}

#constants
VERSION = "1.0b"
BART_MAX_MEM_MB = 4000
PLOTS_DIR = "output_plots"
JAR_DEPENDENCIES = c("bart_java.jar", "commons-math-2.1.jar", "jai_codec.jar", "jai_core.jar", "trove-3.0.3.jar")
DEFAULT_ALPHA = 0.95
DEFAULT_BETA = 2
DEFAULT_K = 2
DEFAULT_Q = 0.9
DEFAULT_NU = 3.0
DEFAULT_PROB_STEPS = c(2.5, 2.5, 4) / 9
DEFAULT_PROB_RULE_CLASS = 0.5
COLORS = array(NA, 500)
for (i in 1 : 500){
	COLORS[i] = rgb(runif(1, 0, 0.7), runif(1, 0, 0.7), runif(1, 0, 0.7))
}

BART_NUM_CORES = 1

set_bart_machine_num_cores = function(num_cores){
	assign("BART_NUM_CORES", num_cores, ".GlobalEnv")
}

init_java_for_bart = function(){
	jinit_params = paste("-Xmx", BART_MAX_MEM_MB, "m", sep = "")
	.jinit(parameters = jinit_params)
	for (dependency in JAR_DEPENDENCIES){
		.jaddClassPath(dependency)
	}	
}

build_bart_machine = function(X, y, 
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
		fix_seed = FALSE,
		run_in_sample = TRUE,
		s_sq_y = "mse", # "mse" or "var"
		unique_name = "unnamed",
		print_tree_illustrations = FALSE,
		num_cores = NULL,
		cov_prior_vec = NULL,
		verbose = TRUE){
	
	t0 = Sys.time()
	#immediately initialize Java
	init_java_for_bart()
	
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
	} else { #otherwise throw an error
		stop("Your response must be either numeric or a factor with two levels.\n")
	}	
	
	num_gibbs = num_burn_in + num_iterations_after_burn_in
	#check for errors in data
	if (check_for_errors_in_training_data(X)){
		return;
	}
	
	model_matrix_training_data = cbind(pre_process_training_data(X), y_numeric)
	
	
	#first set the name
	.jcall(java_bart_machine, "V", "setUniqueName", unique_name)
	#now set whether we want the program to log to a file
	if (debug_log & verbose){
		cat("warning: printing out the log file will slow down the runtime significantly\n")
		.jcall(java_bart_machine, "V", "writeStdOutToLogFile")
	}
	#fix seed if you want
	if (fix_seed){
		.jcall(java_bart_machine, "V", "fixRandSeed")		
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
			mod = lm(y_trans ~ ., as.data.frame(model_matrix_training_data)[1 : (ncol(model_matrix_training_data) - 1)])
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

	
	#make bart to spec with what the user wants
	.jcall(java_bart_machine, "V", "setNumCores", as.integer(ifelse(is.null(num_cores), BART_NUM_CORES, num_cores))) #this must be set FIRST!!!
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
	
	
	if (length(cov_prior_vec) != 0){
		#put in checks here for user to make sure the covariate prior vec is the correct length
		if (length(cov_prior_vec) != ncol(model_matrix_training_data) - 1){
			attribute_names = paste(colnames(model_matrix_training_data)[1 : ncol(model_matrix_training_data) - 1], collapse = ", ")
			stop(paste("covariate prior vector length =", length(cov_prior_vec), "has to be equal to p =", ncol(model_matrix_training_data) - 1, "\nattribute names in order for the prior:", attribute_names), call. = FALSE)
			return(TRUE)
		} else if (sum(cov_prior_vec > 0) != ncol(model_matrix_training_data) - 1){
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
	
	#build the bart machine!
	.jcall(java_bart_machine, "V", "Build")
	
	#now once it's done, let's extract things that are related to diagnosing the build of the BART model
	p = ncol(model_matrix_training_data) - 1
	bart_machine = list(java_bart_machine = java_bart_machine,
		training_data_features = colnames(model_matrix_training_data)[1 : p],
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
			bart_machine$Rsq = 1 - bart_machine$L2_err_train / sum((y - mean(y))^2) #1 - SSE / SST
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
			rownames(confusion_matrix) = c(y_levels, "use errors")
			colnames(confusion_matrix) = c(y_levels, "model errors")
			
			confusion_matrix[1 : 2, 1 : 2] = as.integer(table(y, y_hat_train)) 
			confusion_matrix[3, 1] = round(confusion_matrix[1, 2] / (confusion_matrix[1, 1] + confusion_matrix[2, 1]), 3)
			confusion_matrix[3, 2] = round(confusion_matrix[2, 1] / (confusion_matrix[1, 2] + confusion_matrix[2, 2]), 3)
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
		num_cores = bart_machine$num_cores,
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

size_of_bart_chisq_cache_inquire = function(){
	init_java_for_bart()
	num_bytes = .jcall("CGM_BART.StatToolbox", "I", "numBytesForInvChisqCache")
	cat(paste(round(num_bytes / 1000000, 2), "MB\n"))
	invisible(num_bytes)
}

delete_bart_chisq_cache = function(){
	init_java_for_bart()
	.jcall("CGM_BART.StatToolbox", "V", "clearInvChisqHash")
}

get_var_counts_over_chain = function(bart_machine, type = "splits"){
	C = t(sapply(.jcall(bart_machine$java_bart_machine, "[[I", "getCountsForAllAttribute", as.integer(BART_NUM_CORES), type), .jevalArray))
	colnames(C) = colnames(bart_machine$model_matrix_training_data)[1 : bart_machine$p]
	C
}

get_var_props_over_chain = function(bart_machine, type = "splits"){
	C = get_var_counts_over_chain(bart_machine)	
	Ctot = apply(C, 2, sum)
	Ctot / sum(Ctot)
}

bart_predict_for_test_data = function(bart_machine, X, y){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	y_hat = predict(bart_machine, X)
	n = nrow(X)
	L2_err = sum((y - y_hat)^2)
	
	list(
		y_hat = y_hat,
		L1_err = sum(abs(y - y_hat)),
		L2_err = L2_err,
		rmse = sqrt(L2_err / n),
		e = y - y_hat
	)
}


#
#do all generic functions here
#

bart_machine_predict = function(bart_machine, X){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	#pull out data objects for convenience
	java_bart_machine = bart_machine$java_bart_machine
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	n = nrow(X)
		
	#check for errors in data
	#
	#now process and make dummies if necessary
	X = pre_process_new_data(X, bart_machine$training_data_features)	
		
	y_hat_posterior_samples = 
		t(sapply(.jcall(bart_machine$java_bart_machine, "[[D", "getGibbsSamplesForPrediction", .jarray(X, dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))

	#to get y_hat.. just take straight mean of posterior samples, alternatively, we can let java do it if we want more bells and whistles
	y_hat = rowMeans(y_hat_posterior_samples)	
	
	list(y_hat = y_hat, X = X, y_hat_posterior_samples = y_hat_posterior_samples)
}

predict.bart_machine = function(bart_machine, new_data){
	bart_machine_predict(bart_machine, new_data)$y_hat
}

sigsq_est = function(bart_machine){
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")	
	sigsqs_after_burnin = sigsqs[(length(sigsqs) - bart_machine$num_iterations_after_burn_in) : length(sigsqs)]	
	mean(sigsqs_after_burnin)
}

summary.bart_machine = function(bart_machine, show_details_for_trees = FALSE){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	cat(paste("Bart Machine v", VERSION, ifelse(bart_machine$pred_type == "regression", " for regression", " for classification"), "\n\n", sep = ""))
	#first print out characteristics of the training data
	cat(paste("training data n =", bart_machine$n, " p =", bart_machine$p, "\n"))
	
	ttb = as.numeric(bart_machine$time_to_build, units = "secs")
	if (ttb > 60){
		ttb = as.numeric(bart_machine$time_to_build, units = "mins")
		cat(paste("built in", round(ttb, 2), "mins on", bart_machine$num_cores, "cores,", bart_machine$num_trees, "trees,", bart_machine$num_burn_in, "burn in and", bart_machine$num_iterations_after_burn_in, "posterior samples\n"))
	} else {
		cat(paste("built in", round(ttb, 1), "secs on", bart_machine$num_cores, "cores,", bart_machine$num_trees, "trees,", bart_machine$num_burn_in, "burn in and", bart_machine$num_iterations_after_burn_in, "posterior samples\n"))
	}
	
	if (bart_machine$pred_type == "regression"){
		sigsq_est = sigsq_est(bart_machine)
		cat(paste("\nsigsq est for y beforehand:", round(bart_machine$sig_sq_est, 3), "\n"))
		cat(paste("avg sigsq estimate after burn-in:", round(sigsq_est, 5), "\n"))
		
		if (bart_machine$run_in_sample){
			cat("\nin-sample statistics:\n")
			cat(paste("  L1 = ", round(bart_machine$L1_err_train, 2), 
							"L2 = ", round(bart_machine$L2_err_train, 2),
							"rmse =", round(bart_machine$rmse_train, 2), "\n"))
			
			es = bart_machine$residuals
			normal_p_val = shapiro.test(es)$p.value
			cat("\np-val for shapiro-wilk test of normality of residuals:", round(normal_p_val, 5), "\n")
			
			centered_p_val = t.test(es)$p.value
			cat("p-val for zero-mean noise:", round(centered_p_val, 5), "\n")	
		} else {
			cat("\nno in-sample information available (use option run_in_sample = TRUE next time)\n")
		}		
	} else if (bart_machine$pred_type == "classification"){
		if (bart_machine$run_in_sample){
			cat("\nconfusion matrix:\n\n")
			print(bart_machine$confusion_matrix)
		} else {
			cat("\nno in-sample information available (use option run_in_sample = TRUE next time)\n")
		}		
	}

	
	if (show_details_for_trees){
		cat("\nproportion M-H steps accepted:\n")	
		cat(paste("  before burn-in:", round(0, 2), "after burn-in:", round(0, 2), "overall:", round(0, 2), "\n"))
		
		cat(paste("\nquantiles of tree depths after burn in:\n"))
		tree_depths = rnorm(1000)
		print(round(summary(tree_depths), 2))
		cat(paste("quantiles of number of splits after burn in:\n"))
		tree_splits = rnorm(1000)
		print(round(summary(tree_splits), 2))		
	}
	cat("\n")
}

plot.bart_machine = function(bart_machine, plot_list = c("a", "b", "c", "d", "e")){
	#plot 1
	windows()
	#plot 2
	windows()
	#plot 3
	#etc
}

print.bart_machine = function(bart_machine){
	summary(bart_machine)
}

calc_ppis_from_prediction = function(bart_machine, new_data, ppi_conf = 0.95){
	#first convert the rows to the correct dummies etc
	new_data = pre_process_new_data(new_data, bart_machine$training_data_features)
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

check_for_errors_in_training_data = function(data){
	if (is.null(colnames(data))){
		stop("no colnames in data matrix", call. = FALSE)
		return(TRUE)
	}
	if (class(data) != "data.frame"){
		stop(paste("training data must be a data frame. It is currently a", class(data)), call. = FALSE)
		return(TRUE)		
	}
	FALSE
}

pre_process_training_data = function(data){
	#delete missing data just in case
	data = na.omit(data)
	
	#first convert characters to factors
	character_vars = names(which(sapply(data, class) == "character"))
	for (character_var in character_vars){
		data[, character_var] = as.factor(data[, character_var])
	}
		
	factors = names(which(sapply(data, class) == "factor"))
	
	for (fac in factors){
		dummied = do.call(cbind, lapply(levels(data[, fac]), function(lev){as.numeric(data[, fac] == lev)}))
		colnames(dummied) <- paste(fac, levels(data[, fac]), sep = "_")		
		data = cbind(data, dummied)
		data[, fac] = NULL
	}
	
	data.matrix(data)
}

pre_process_new_data = function(new_data, training_data_features){
	new_data = pre_process_training_data(new_data)
	n = nrow(new_data)
	new_data_features = colnames(new_data)
	
	#iterate through and see
	for (j in 1 : length(training_data_features)){
		training_data_feature = training_data_features[j]
		new_data_feature = new_data_features[j]
		if (training_data_feature != new_data_feature){
			#create a new col of zeroes
			new_col = rep(0, n)
			#wedge it into the data set
			temp_new_data = cbind(new_data[, 1 : (j - 1)], new_col)
			#give it the same name as in the training set
			colnames(temp_new_data)[j] = training_data_feature
			#tack on the rest of the stuff
			if (ncol(new_data) >= j){
				rhs = new_data[, j : ncol(new_data)]
				if (class(rhs) == "numeric"){
					rhs = as.matrix(rhs)
					colnames(rhs)[1] = new_data_feature
				}
				temp_new_data = cbind(temp_new_data, rhs)
			} 
			new_data = temp_new_data

			#update list
			new_data_features = colnames(new_data)
		}
	}
	#coerce to numeric
	data.matrix(new_data)
}

#believe it or not... there's no standard R function for this, isn't that pathetic?
sample_mode = function(data){
	as.numeric(names(sort(-table(data)))[1])
}


#set up a logging system
LOG_DIR = "r_log"
log_file_name = "bart_log.txt"
bart_log = matrix(nrow = 0, ncol = 1)


append_to_log = function(text){
	bart_log = rbind(bart_log, paste(Sys.time(), "\t", text)) #log the time and the actual message
	assign("bart_log", bart_log, .GlobalEnv)
	write.table(bart_log, paste(LOG_DIR, "/", log_file_name, sep = ""), quote = FALSE, col.names = FALSE, row.names = FALSE)
}