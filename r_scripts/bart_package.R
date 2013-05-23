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
BART_MAX_MEM_MB = 8000
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

set_bart_machine_max_mem = function(max_mem_mbs){
	assign("BART_MAX_MEM_MB", max_mem_mbs, ".GlobalEnv")
}

BART_NUM_CORES = 1

set_bart_machine_num_cores = function(num_cores){
	assign("BART_NUM_CORES", num_cores, ".GlobalEnv")
}

bart_machine_num_cores = function(){
	BART_NUM_CORES
}

init_java_for_bart = function(){
	jinit_params = paste("-Xmx", BART_MAX_MEM_MB, "m", sep = "")
	.jinit(parameters = jinit_params)
	for (dependency in JAR_DEPENDENCIES){
		.jaddClassPath(dependency)
	}	
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
	attribute_props = .jcall(bart_machine$java_bart_machine, "[D", "getAttributeProps", as.integer(BART_NUM_CORES), type)
	names(attribute_props) = colnames(bart_machine$model_matrix_training_data)[1 : bart_machine$p]
	attribute_props
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

bart_machine_predict = function(bart_machine, X, ppi = 0.95){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	if (class(X) != "matrix" && class(X) != "data.frame"){		
		stop("X needs to be a matrix or data frame with the same column names as the training data.")
	}
#	if (sum(is.na(X)) == length(X)){
#		stop("Cannot predict on all missing data.\n")
#	}
	if (!bart_machine$use_missing_data){
		nrow_before = nrow(X)
		X = na.omit(X)
		if (nrow_before > nrow(X)){
			cat(nrow_before - nrow(X), "rows omitted due to missing data\n")
		}
	}
	
	if (nrow(X) == 0){
		stop("No rows to predict.\n")
	}
	#pull out data objects for convenience
	java_bart_machine = bart_machine$java_bart_machine
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	n = nrow(X)
		
	#check for errors in data
	#
	#now process and make dummies if necessary
	X = pre_process_new_data(X, bart_machine)
		
	#check for missing data if this feature was not turned on
	if (!bart_machine$use_missing_data){
		M = matrix(0, nrow = nrow(X), ncol = ncol(X))
		for (i in 1 : nrow(X)){
			for (j in 1 : ncol(X)){
				if (is.missing(X[i, j])){
					M[i, j] = 1
				}
			}
		}
		if (sum(M) > 0){
			cat("WARNING: missing data found in test data and BART was not built with missing data feature!\n")
		}		
	}
	
	y_hat_posterior_samples = 
		t(sapply(.jcall(bart_machine$java_bart_machine, "[[D", "getGibbsSamplesForPrediction", .jarray(X, dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))

	#to get y_hat.. just take straight mean of posterior samples, alternatively, we can let java do it if we want more bells and whistles
	y_hat = rowMeans(y_hat_posterior_samples)	
	
	ppi_a = apply(y_hat_posterior_samples, 1, quantile, probs = (1 - ppi) / 2)
	ppi_b = apply(y_hat_posterior_samples, 1, quantile, probs = ppi + (1 - ppi) / 2)
	
	list(y_hat = y_hat, X = X, y_hat_posterior_samples = y_hat_posterior_samples, ppi_a = ppi_a, ppi_b = ppi_b)
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
	if (bart_machine$use_missing_data){
		cat("Missing data feature ON\n")
	}
	#first print out characteristics of the training data
	cat(paste("training data n =", bart_machine$n, "and p =", bart_machine$p, "\n"))
	
	ttb = as.numeric(bart_machine$time_to_build, units = "secs")
	if (ttb > 60){
		ttb = as.numeric(bart_machine$time_to_build, units = "mins")
		cat(paste("built in", round(ttb, 2), "mins on", bart_machine$num_cores, ifelse(bart_machine$num_cores == 1, "core,", "cores,"), bart_machine$num_trees, "trees,", bart_machine$num_burn_in, "burn in and", bart_machine$num_iterations_after_burn_in, "posterior samples\n"))
	} else {
		cat(paste("built in", round(ttb, 1), "secs on", bart_machine$num_cores, "cores,", bart_machine$num_trees, "trees,", bart_machine$num_burn_in, "burn in and", bart_machine$num_iterations_after_burn_in, "posterior samples\n"))
	}
	
	if (bart_machine$pred_type == "regression"){
		sigsq_est = sigsq_est(bart_machine)
		cat(paste("\nsigsq est for y beforehand:", round(bart_machine$sig_sq_est, 3), "\n"))
		cat(paste("avg sigsq estimate after burn-in:", round(sigsq_est, 5), "\n"))
		
		if (bart_machine$run_in_sample){
			cat("\nin-sample statistics:\n")
			cat(paste(" L1 =", round(bart_machine$L1_err_train, 2), "\n",
					   "L2 =", round(bart_machine$L2_err_train, 2), "\n",
					   "rmse =", round(bart_machine$rmse_train, 2), "\n"),
					   "Pseudo-Rsq =", round(bart_machine$PseudoRsq, 4))
			
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
	new_data = pre_process_new_data(new_data, bart_machine)
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
	if (class(data) != "data.frame"){
		stop(paste("training data must be a data frame. It is currently a", class(data)), call. = FALSE)
		return(TRUE)		
	}
	FALSE
}

pre_process_training_data = function(data, use_missing_data = TRUE, verbose = FALSE){
	
	#first convert characters to factors
	character_vars = names(which(sapply(data, class) == "character"))
	for (character_var in character_vars){
		data[, character_var] = as.factor(data[, character_var])
	}
		
	factors = names(which(sapply(data, class) == "factor"))
	
	for (fac in factors){
		dummied = do.call(cbind, lapply(levels(data[, fac]), function(lev){as.numeric(data[, fac] == lev)}))
		colnames(dummied) = paste(fac, levels(data[, fac]), sep = "_")		
		data = cbind(data, dummied)
		data[, fac] = NULL
	}
	
	if (use_missing_data){		
		#now take care of missing data
		M = matrix(0, nrow = nrow(data), ncol = ncol(data))
		for (i in 1 : nrow(data)){
			for (j in 1 : ncol(data)){
				if (is.missing(data[i, j])){
					M[i, j] = 1
				}
			}
		}
		colnames(M) = paste("M_", colnames(data), sep = "")
		#append the missing dummy columns to data as if they're real attributes themselves
		data = cbind(data, M)
	}
	#make sure to cast it as a data matrix
	data.matrix(data)
}

is.missing = function(x){
	is.na(x) || is.nan(x)
}

###TO-DO this has to updated for the M matrix
pre_process_new_data = function(new_data, bart_machine){
	new_data = as.data.frame(new_data)
	new_data = pre_process_training_data(new_data, bart_machine$use_missing_data, bart_machine$verbose)
	n = nrow(new_data)
	new_data_features = colnames(new_data)
	
	if (bart_machine$use_missing_data){
		training_data_features = bart_machine$training_data_features_with_missing_features
	} else {
		training_data_features = bart_machine$training_data_features
	}
	
	
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
	#coerce to a numeric matrix
	new_data = data.matrix(new_data)
	mode(new_data) = "numeric"
	new_data
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