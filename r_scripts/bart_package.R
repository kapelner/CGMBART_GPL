
#libraries and dependencies
options(repos = "http://lib.stat.cmu.edu/R/CRAN")
tryCatch(library(randomForest), error = function(e){install.packages("randomForest")}, finally = library(randomForest))
tryCatch(library(rpart), error = function(e){install.packages("rpart")}, finally = library(rpart))
tryCatch(library(xtable), error = function(e){install.packages("xtable")}, finally = library(xtable))
tryCatch(library(rJava), error = function(e){install.packages("rJava")}, finally = library(rJava))
tryCatch(library(BayesTree), error = function(e){install.packages("BayesTree")}, finally = library(BayesTree))
#
#if (.Platform$OS.type == "windows"){
#	tryCatch(library(rJava), error = function(e){install.packages("rJava")}, finally = library(rJava))
#	tryCatch(library(BayesTree), error = function(e){install.packages("BayesTree")}, finally = library(BayesTree))
#} else {
#	tryCatch(library(rJava), error = function(e){library(rJava, lib.loc = "~/R/")})	
#	library(BayesTree, lib.loc = "~/R/")
#}

#constants
NUM_MEGS_RAM_TO_USE = 6000 #ifelse(.Platform$OS.type == "windows", 6000, 1600) #1690
PLOTS_DIR = "output_plots"
JAR_DEPENDENCIES = c("bart_java.jar", "commons-math-2.1.jar", "jai_codec.jar", "jai_core.jar", "trove-3.0.3.jar")
DATA_FILENAME = "datasets/bart_data.csv"
DEFAULT_ALPHA = 0.95
DEFAULT_BETA = 2
COLORS = array(NA, 500)
for (i in 1 : 500){
	COLORS[i] = rgb(runif(1, 0, 0.7), runif(1, 0, 0.7), runif(1, 0, 0.7))
}

#immediately initialize Java
jinit_params = paste("-Xmx", NUM_MEGS_RAM_TO_USE, "m", sep = "")
.jinit(parameters = jinit_params)


#set up a logging system
LOG_DIR = "r_log"
log_file_name = "bart_log.txt"
bart_log = matrix(nrow = 0, ncol = 1)


append_to_log = function(text){
	bart_log = rbind(bart_log, paste(Sys.time(), "\t", text)) #log the time and the actual message
	assign("bart_log", bart_log, .GlobalEnv)
	write.table(bart_log, paste(LOG_DIR, "/", log_file_name, sep = ""), quote = FALSE, col.names = FALSE, row.names = FALSE)
}

#some defaults if you want to run this
num_trees = 1
num_burn_in = 1000
num_iterations_after_burn_in = 1000
num_gibbs = num_burn_in + num_iterations_after_burn_in

class_or_regr = "r"
debug_log = TRUE
print_tree_illustrations = FALSE
PRINT_TREE_ILLUS = FALSE
print_out_every = NULL
fix_seed = FALSE
JAVA_LOG = FALSE #to be overwritten later
#source("r_scripts/create_simulated_models.R")
#simulated_data_model_name = simulated_data_sets[1]
#training_data = simulate_data_from_simulation_name(simulated_data_model_name)
#test_data = simulate_data_from_simulation_name(simulated_data_model_name)

build_bart_machine = function(training_data, 
		num_trees = 200, 
		num_burn_in = 2000, 
		num_iterations_after_burn_in = 2000, 
		alpha = DEFAULT_ALPHA,
		beta = DEFAULT_BETA, 
		regression_type = "r", #r, c, o, n (regression, classification, ordinal, count)
		debug_log = FALSE,
		fix_seed = FALSE,
		run_in_sample = TRUE,
		s_sq_y = "mse", #"mse" or "var"
		unique_name = "unnamed",
		print_tree_illustrations = FALSE,
		num_cores = 1,
		use_heteroskedasticity = FALSE,
		cov_prior_vec = NULL){
	
	num_gibbs = num_burn_in + num_iterations_after_burn_in
	#check for errors in data
	if (error_in_data(training_data)){
		return;
	}
	model_matrix_training_data = pre_process_data(training_data)
	
	#now write the data as a csv, make sure to exclude row names,
	#this is how we're going to send the data in BART for now even
	#though it is inefficient
	write.csv(model_matrix_training_data, DATA_FILENAME, row.names = FALSE)
	
	#initialize the JVM
	for (dependency in JAR_DEPENDENCIES){
		.jaddClassPath(paste(directory_where_code_is, "/", dependency, sep = ""))
	}
	java_bart_machine = .jnew("CGM_BART.CGMBARTRegressionMultThread")
	
	#first set the name
	.jcall(java_bart_machine, "V", "setUniqueName", unique_name)
	#now set whether we want the program to log to a file
	if (debug_log){
		cat("warning: printing out the log file will slow down the runtime significantly\n")
		.jcall(java_bart_machine, "V", "writeStdOutToLogFile")
	}
	#fix seed if you want
	if (fix_seed){
		.jcall(java_bart_machine, "V", "fixRandSeed")		
	}
	#set whether we want there to be tree illustrations
	if (print_tree_illustrations){
		cat("warning: printing tree illustrations will slow down the runtime significantly\n")
		.jcall(java_bart_machine, "V", "printTreeIllustations")
	}
	
	#set the std deviation of y to use
	if (ncol(model_matrix_training_data) - 1 >= nrow(model_matrix_training_data)){
		cat("warning: cannot use MSE of linear model for s_sq_y if p > n\n")
		s_sq_y = "var"
	}
	if (s_sq_y == "mse"){
		mod = lm(y ~ ., training_data)
		mse = var(mod$residuals)
		.jcall(java_bart_machine, "V", "setSampleVarY", as.numeric(mse))
	} else if (s_sq_y == "var"){
		.jcall(java_bart_machine, "V", "setSampleVarY", as.numeric(var(model_matrix_training_data$y)))
	} else {
		stop("s_sq_y must be \"rmse\" or \"sd\"", call. = FALSE)
		return(TRUE)
	}
	
	#make bart to spec with what the user wants
	.jcall(java_bart_machine, "V", "setNumCores", as.integer(num_cores)) #this must be set FIRST!!!
	.jcall(java_bart_machine, "V", "setNumTrees", as.integer(num_trees))
	.jcall(java_bart_machine, "V", "setNumGibbsBurnIn", as.integer(num_burn_in))
	.jcall(java_bart_machine, "V", "setNumGibbsTotalIterations", as.integer(num_gibbs))
	.jcall(java_bart_machine, "V", "setAlpha", alpha)
	.jcall(java_bart_machine, "V", "setBeta", beta)

	
	
	if (length(cov_prior_vec) != 0){
		#put in checks here for user to make sure it's correct length
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
	if (use_heteroskedasticity){
		.jcall(java_bart_machine, "V", "useHeteroskedasticity")
	}
	
	#now load the data into BART
	.jcall(java_bart_machine, "V", "setDataToDefaultForRPackage")
	
	#build the bart machine for use later
	#need http://math.acadiau.ca/ACMMaC/Rmpi/sample.html RMPI here
	#or better yet do this in pieces and plot slowly
	.jcall(java_bart_machine, "V", "Build")
	
	#now once it's done, let's extract things that are related to diagnosing the build of the BART model
	bart_machine = list(java_bart_machine = java_bart_machine, 
		training_data = model_matrix_training_data,
		n = nrow(model_matrix_training_data),
		p = ncol(model_matrix_training_data) - 1,
		num_trees = num_trees,
		num_burn_in = num_burn_in,
		num_iterations_after_burn_in = num_iterations_after_burn_in, 
		num_gibbs = num_gibbs,
		alpha = alpha,
		beta = beta,
		run_in_sample = run_in_sample,
		cov_prior_vec = cov_prior_vec,
		use_heteroskedasticity = use_heteroskedasticity
	)
	
	#once its done gibbs sampling, see how the training data does if user wants
	if (run_in_sample){
		cat("evaluating in sample data")
		y_hat_train = array(NA, nrow(model_matrix_training_data))
		for (i in 1 : nrow(model_matrix_training_data)){
			if (i %% as.integer(nrow(model_matrix_training_data) / 10) == 0){
				cat(".")
			}
			y_hat_train[i] = .jcall(java_bart_machine, "D", "Evaluate", c(as.numeric(model_matrix_training_data[i, ])), as.integer(num_cores))
		}
		
		bart_machine$y_hat_train = y_hat_train
		bart_machine$residuals = training_data$y - bart_machine$y_hat_train
		bart_machine$L1_err_train = sum(abs(bart_machine$residuals))
		bart_machine$L2_err_train = sum(bart_machine$residuals^2)
		bart_machine$Rsq = 1 - bart_machine$L2_err_train / sum((training_data$y - mean(training_data$y))^2) #1 - SSE / SST
		bart_machine$rmse_train = sqrt(bart_machine$L2_err_train / bart_machine$n)
		cat("done\n")
	}
	
	bart_machine
}

printed_out_warnings = FALSE

check_bart_error_assumptions = function(bart_machine, alpha_normal_test = 0.05, alpha_hetero_test = 0.05){
	graphics.off()
	es = bart_machine$residuals
	qqnorm(es, main = "Normal Q-Q plot for in-sample residuals")
	qqline(bart_machine$residuals)
	normal_p_val = shapiro.test(es)$p.value
	cat("p-val for shapiro-wilk test of normality of residuals:", normal_p_val, ifelse(normal_p_val > alpha_normal_test, "(ppis believable)", "(exercise caution when using ppis!)"), "\n")
	
	if (!bart_machine$use_heteroskedasticity){
		#see p225 in purple book for Szroeter's test
		n = length(es)
		h = sum(seq(1 : n) * es^2) / sum(es^2)
		Q = sqrt(6 * n / (n^2 - 1)) * (h - (n + 1) / 2)
		hetero_pval = 1 - pnorm(Q, 0, 1)
		cat("p-val for szroeter's test of homoskedasticity of residuals (assuming inputted observation order):", hetero_pval, ifelse(hetero_pval > alpha_hetero_test, "(ppis believable)", "(exercise caution when using ppis!)"), "\n")		
	}
}


clean_previous_bart_data = function(){
	if (exists("sigsqs_after_burnin")){
		rm(sigsqs_after_burnin)
	}
	if (exists("all_tree_num_nodes")){
		rm(all_tree_num_nodes)
	}
	if (exists("all_tree_liks")){
		rm(all_tree_liks)
	}
	if (exists("y_hat")){
		rm(y_hat)
	}
	if (exists("all_mu_vals_for_all_trees")){
		rm(all_mu_vals_for_all_trees)
	}
}

plot_sigsqs_convergence_diagnostics_hetero = function(bart_machine, records = c(1), extra_text = NULL, data_title = "data_model", save_plot = FALSE, moving_avgs = TRUE){
	if (bart_machine$use_heteroskedasticity != TRUE){
		stop("This BART machine was not created using the heteroskedasticity-robust feature, use \"plot_sigsqs_convergence_diagnostics\" instead")
	}
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
	
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	sigsqs = matrix(NA, nrow = bart_machine$num_gibbs, ncol = bart_machine$n)
	for (g in 1 : bart_machine$num_gibbs){
		sigsqs[g, ] = .jcall(bart_machine$java_bart_machine, "[D", "getSigsqsByGibbsSample", as.integer(g - 1))
	}
	sigsqs_after_burnin = sigsqs[(length(sigsqs) - num_iterations_after_burn_in) : length(sigsqs), 1]
	assign("sigsqs_after_burnin", sigsqs_after_burnin, .GlobalEnv)	
	
	if (save_plot){
		save_plot_function(bart_machine, "sigsqs_by_gibbs", data_title)
	}
	else {
		dev.new()
	}	
	
	ymax = quantile(sigsqs, .95)
#	ymax = max(sigsqs[bart_machine$num_burn_in : bart_machine$num_gibbs, ])
	
	plot(NA, 
		main = paste("Sigsqs throughout entire chain", ifelse(is.null(extra_text), "", 
						paste("\n", extra_text)), ifelse(length(records) < 10, paste("record #", paste(records, collapse = ", ")), "")), 
		type = "n", 
		xlab = "Gibbs sample",
		ylab = "Var[Noise] estimate",
		xlim = c(1, bart_machine$num_gibbs), 		
		ylim = c(0, ymax)
	)

	#want to plot each sigsq as a function of gibbs
	for (i in records){
		sigsqis = sigsqs[, i]
		points(sigsqis, pch = ".", col = COLORS[i %% 500])
	}
	abline(v = num_burn_in, col = "gray")
	
	#now maybe we want to see moving averages
	if (moving_avgs){
		for (i in records){
			sigsqis = sigsqs[, i]
			moving_avg = filter(sigsqis, rep(1/101, 101), sides = 2)
			lines(moving_avg, col = COLORS[i %% 500])
		}		
	}
	
	if (save_plot){	
		dev.off()
	}
	
	sigsqs
}

plot_sigsqs_convergence_diagnostics = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	if (bart_machine$use_heteroskedasticity == TRUE){
		stop("This BART machine was created using the heteroskedasticity-robust feature, use \"plot_sigsqs_convergence_diagnostics_hetero\" instead")
	}	
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
	
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	#first look at sigsqs
	sigsqs_after_burnin = sigsqs[(length(sigsqs) - num_iterations_after_burn_in) : length(sigsqs)]
	assign("sigsqs_after_burnin", sigsqs_after_burnin, .GlobalEnv)
	avg_sigsqs = mean(sigsqs_after_burnin, na.rm = TRUE)
	 
	if (save_plot){
		save_plot_function(bart_machine, "sigsqs_by_gibbs", data_title)
	}
	else {
		dev.new()
	}
	plot(sigsqs, 
			main = paste("Sigsq throughout entire project  sigsq after burn in", ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
			xlab = "Gibbs sample # (gray line indicates burn-in, yellow lines are the 95% PPI after burn-in)", 
			ylab = paste("Sigsq by iteration, avg after burn-in =", round(avg_sigsqs, 3)),
			ylim = c(quantile(sigsqs, 0.01), quantile(sigsqs, 0.99)),
			pch = ".", 
			cex = 3,
			col = "gray")
	points(sigsqs, pch = ".", col = "red")
	ppi_sigsqs = quantile(sigsqs[num_burn_in : length(sigsqs)], c(.025, .975))
	abline(a = ppi_sigsqs[1], b = 0, col = "yellow")
	abline(a = ppi_sigsqs[2], b = 0, col = "yellow")
	abline(a = avg_sigsqs, b = 0, col = "blue")
	abline(v = num_burn_in, col = "gray")
	
	if (save_plot){	
		dev.off()
	}
	
	sigsqs_after_burnin
}

hist_sigsqs = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
	
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	sigsqs_after_burnin = sigsqs[(length(sigsqs) - num_iterations_after_burn_in) : length(sigsqs)]
	assign("sigsqs_after_burnin", sigsqs_after_burnin, .GlobalEnv)
	avg_sigsqs = mean(sigsqs_after_burnin, na.rm = TRUE)
	
	if (save_plot){
		save_plot_function(bart_machine, "sigsqs_hist", data_title)
	}	
	else {
		dev.new()
	}
	
	ppi_a = quantile(sigsqs_after_burnin, 0.025)
	ppi_b = quantile(sigsqs_after_burnin, 0.975)
	hist(sigsqs_after_burnin, 
		br = 100, 
		main = "Histogram of sigsqs after burn-in", 
		xlab = paste("sigsq  avg = ", round(avg_sigsqs, 1), ",  95% PPI = [", round(ppi_a, 1), ", ", round(ppi_b, 1), "]", sep = ""))
	abline(v = avg_sigsqs, col = "blue")
	abline(v = ppi_a, col = "yellow")
	abline(v = ppi_b, col = "yellow")
	
	if (save_plot){	
		dev.off()
	}
	
	sigsqs_after_burnin
}

maximum_nodes_over_all_trees = function(bart_machine){
	.jcall(bart_machine$java_bart_machine, "I", "maximalNodeNumber")
}

get_mu_values_for_all_trees = function(bart_machine){
	java_bart_machine = bart_machine$java_bart_machine
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	max_b = maximum_nodes_over_all_trees(bart_machine)
	
	all_mu_vals_for_all_trees = array(NA, c(max_b, num_gibbs, num_trees))
	for (t in 1 : num_trees){
		for (b in 1 : max_b){
			doubles = .jcall(java_bart_machine, "[D", "getMuValuesForAllItersByTreeAndLeaf", as.integer(t - 1), as.integer(b))
			doubles[doubles == -9999999] = NA #stupid RJava
			all_mu_vals_for_all_trees[b, , t] = doubles
		}
	}
	assign("all_mu_vals_for_all_trees", all_mu_vals_for_all_trees, .GlobalEnv)
}

hist_mu_values_by_tree_and_leaf_after_burn_in = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE, tree_num, leaf_num){
	get_mu_values_for_all_trees(bart_machine)
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs	
	
	all_mu_vals_for_tree_and_leaf = all_mu_vals_for_all_trees[leaf_num, , tree_num]
	#only after num_burn_in
	all_mu_vals_for_tree_and_leaf = all_mu_vals_for_tree_and_leaf[(num_burn_in + 1) : num_gibbs]

	num_not_nas = length(all_mu_vals_for_tree_and_leaf) - sum(is.na(all_mu_vals_for_tree_and_leaf))
	
	if (num_not_nas == 0){
#		cat("WARNING: This node was never a leaf\n")
		return()
	}
	
	if (save_plot){
		save_plot_function(bart_machine, paste("mu_vals_t_", tree_num, "_b_", leaf_num, sep = ""), data_title)
	}	
	else {
		dev.new()
	}
	avg_mu_for_leaf = mean(all_mu_vals_for_tree_and_leaf)
	ppi_a = quantile(all_mu_vals_for_tree_and_leaf, 0.025, na.rm = TRUE)
	ppi_b = quantile(all_mu_vals_for_tree_and_leaf, 0.975, na.rm = TRUE)
	
	hist(all_mu_vals_for_tree_and_leaf,
		br = 100, 
		main = paste("Mu Values for tree ", tree_num, " leaf ", leaf_num, " for the ", num_not_nas, " mu's after burn in (when a leaf)", ifelse(is.null(extra_text), "", paste("\n", extra_text)), sep = ""), 
		xlab = paste("Mu values avg = ", round(avg_mu_for_leaf, 2), ",  95% PPI = [", round(ppi_a, 2), ",", round(ppi_b, 2), "]", sep = ""), 
		ylab = "value",
		col = COLORS[leaf_num],
		border = NA)
	abline(v = avg_mu_for_leaf, col = "blue", lwd = 4)
	abline(v = ppi_a, col = "yellow", lwd = 3)
	abline(v = ppi_b, col = "yellow", lwd = 3)
	
	if (save_plot){
		dev.off()
	}
}

plot_all_mu_values_for_tree = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE, tree_num){
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs	
	
	get_mu_values_for_all_trees(bart_machine)
	
	all_mu_vals_for_tree = all_mu_vals_for_all_trees[, , tree_num]
	
	if (save_plot){
		save_plot_function(bart_machine, paste("plot_mu_vals_t_", tree_num, sep = ""), data_title)
	}	
	else {
		dev.new()
	}
	plot(1 : num_gibbs, 
			NULL,  # + 1 for the prior
			type = "n", 
			main = paste("Mu Values for all leaves in tree ", tree_num, ifelse(is.null(extra_text), "", paste("\n", extra_text)), sep = ""), 
			ylim = c(min(all_mu_vals_for_tree, na.rm = TRUE), max(all_mu_vals_for_tree, na.rm = TRUE)),
			xlab = "Gibbs sample # (gray line indicates burn in)", 
			ylab = "value")
	for (b in 1 : maximum_nodes_over_all_trees(bart_machine)){
		points(1 : num_gibbs, all_mu_vals_for_tree[b, ], col = COLORS[b], pch = ".", type = "l", cex = 2)
	}
	abline(v = num_burn_in, col = "gray")
	
	if (save_plot){
		dev.off()
	}
}

hist_all_mu_values_for_tree = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE, tree_num){
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs	
	
	get_mu_values_for_all_trees(bart_machine)
	
	all_mu_vals_for_tree_after_burn_in = all_mu_vals_for_all_trees[, num_burn_in : num_gibbs, tree_num]
	
	if (save_plot){
		save_plot_function(bart_machine, paste("hist_mu_vals_t_", tree_num, sep = ""), data_title)
	}	
	else {
		dev.new()
	}
	
	all_mu_vals_for_tree_vec_after_burn_in = as.vector(all_mu_vals_for_tree_after_burn_in)
	min_mu = min(all_mu_vals_for_tree_vec_after_burn_in, na.rm = TRUE)
	max_mu = max(all_mu_vals_for_tree_vec_after_burn_in, na.rm = TRUE)
	
	hist(all_mu_vals_for_tree_vec_after_burn_in, 
			col = "white",
			border = NA, 
			br = seq(from = min_mu - 0.01, to = max_mu + 0.01, by = (max_mu - min_mu + 0.01) / 5000),
			xlab = "Leaf value",
			main = paste("Hist of all mu values for all leaves in tree ", tree_num, " after burn-in", ifelse(is.null(extra_text), "", paste("\n", extra_text)), sep = ""))
	for (b in 1 : maximum_nodes_over_all_trees(bart_machine)){
		hist(all_mu_vals_for_tree_after_burn_in[b, ], 
			col = COLORS[b], 
			border = NA, 
			br = seq(from = min_mu - 0.01, to = max_mu + 0.01, by = (max_mu - min_mu + 0.01) / 2000), 
			add = TRUE)
	}
	abline(v = num_burn_in, col = "gray")
	
	if (save_plot){
		dev.off()
	}
}

plot_tree_num_nodes = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	java_bart_machine = bart_machine$java_bart_machine
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	all_tree_num_nodes = matrix(NA, nrow = num_trees, ncol = num_gibbs + 1)
	for (n in 1 : (num_gibbs + 1)){
		all_tree_num_nodes[, n] = .jcall(java_bart_machine, "[I", "getNumNodesForTreesInGibbsSamp", as.integer(n - 1))
	}
	assign("all_tree_num_nodes", all_tree_num_nodes, .GlobalEnv)
	
	if (save_plot){
		save_plot_function(bart_machine, "tree_nodes", data_title)
	}	
	else {
		dev.new()
	}	
	plot(1 : (num_gibbs + 1),  # + 1 for the prior
			all_tree_num_nodes[1, ], 
			col = sample(COLORS, 1),
			pch = ".",
			type = "l", 
			main = paste("Num Nodes by Iteration", ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
			ylim = c(1, max(all_tree_num_nodes)),
			xlab = "Gibbs sample # (gray line indicates burn in)", 
			ylab = "num nodes")
	if (num_trees > 1){
		for (t in 2 : nrow(all_tree_num_nodes)){
			points(1 : (num_gibbs + 1), all_tree_num_nodes[t, ], col = sample(COLORS, 1), pch = ".", type = "l", cex = 2)
		}
	}
	abline(v = num_burn_in, col = "gray")
	
	if (save_plot){	
		dev.off()
	}
}

get_root_splits_of_trees = function(bart_machine, data_title = "data_model", save_as_csv = FALSE){
	java_bart_machine = bart_machine$java_bart_machine
	num_gibbs = bart_machine$num_gibbs
	num_burn_in = bart_machine$num_burn_in
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	num_trees = bart_machine$num_trees
	
	root_splits = matrix(NA, nrow = num_gibbs + 1, ncol = 2) #column vector
	colnames(root_splits) = c("gibbs_iter", "root_split_as_string")
	for (n in 1 : (num_gibbs + 1)){
		root_splits[n, 1] = n - 1
		root_splits[n, 2] = .jcall(java_bart_machine, "S", "getRootSplits", as.integer(n - 1))
	}	
	assign("root_splits", root_splits, .GlobalEnv)
	if (save_as_csv){
		csv_filename = paste(PLOTS_DIR, "/", data_title, "_first_splits_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, ".csv", sep = "")
		write.csv(root_splits, csv_filename, row.names = FALSE)
	}
	root_splits
}

get_var_counts_over_chain = function(bart_machine, num_cores_count = 1){
	C = matrix(NA, nrow = bart_machine$num_iterations_after_burn_in, ncol = bart_machine$p)
	for (g in 1 : bart_machine$num_iterations_after_burn_in){
		C[g, ] = .jcall(bart_machine$java_bart_machine, "[I", "getCountForAttributeInGibbsSample", as.integer(g - 1), as.integer(num_cores_count))
	}
	C
}

plot_tree_depths = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	java_bart_machine = bart_machine$java_bart_machine
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	all_tree_depths = matrix(NA, nrow = num_trees, ncol = num_gibbs + 1)
	
	for (n in 1 : (num_gibbs + 1)){
		all_tree_depths[, n] = .jcall(java_bart_machine, "[I", "getDepthsForTreesInGibbsSamp", as.integer(n - 1))
	}
	assign("all_tree_depths", all_tree_depths, .GlobalEnv)
	
	if (save_plot){
		save_plot_function(bart_machine, "tree_depths", data_title)
	}	
	else {
		dev.new()
	}	
	plot(1 : (num_gibbs + 1),  # + 1 for the prior
			all_tree_depths[1, ], 
			col = rgb(runif(1, 0, 0.7), runif(1, 0, 0.7), runif(1, 0, 0.7)),
			pch = ".",
			type = "l", 
			main = paste("Depth by Iteration", ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
			ylim = c(1, max(all_tree_depths)),
			xlab = "Gibbs sample # (gray line indicates burn in)", 
			ylab = "depth")
	if (num_trees > 1){
		for (t in 2 : nrow(all_tree_depths)){
			points(1 : (num_gibbs + 1), all_tree_depths[t, ], col = sample(COLORS, 1), pch = ".", type = "l", cex = 2)
		}
	}
	abline(v = num_burn_in, col = "gray")
	
	if (save_plot){	
		dev.off()
	}
}

ALPHA = 0.5

plot_tree_liks_convergence_diagnostics = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	java_bart_machine = bart_machine$java_bart_machine
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	all_tree_liks = matrix(NA, nrow = num_trees, ncol = num_gibbs + 1)	
	for (t in 1 : num_trees){		
#		tryCatch({
		all_tree_liks[t, ] = .jcall(java_bart_machine, "[D", "getLikForTree", as.integer(t - 1))
#		},
#		error = function(exc){return},
#		finally = function(exc){})			
	}	
	assign("all_tree_liks", all_tree_liks, .GlobalEnv)
	
	treeliks_scale = (max(all_tree_liks, na.rm = TRUE) - min(all_tree_liks, na.rm = TRUE)) * 0.5
	
	if (save_plot){
		save_plot_function(bart_machine, "tree_liks_by_gibbs", data_title)
	}	
	else {
		dev.new()
	}	
	plot(1 : (num_gibbs + 1),  # + 1 for the prior
			all_tree_liks[1, ], 
			col = sample(COLORS, 1),
			pch = ".",
			main = paste("Tree ln(prop Lik) by Iteration", ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
			ylim = quantile(all_tree_liks[1, ], c(0, .999), na.rm = TRUE),
			xlab = "Gibbs sample # (gray line indicates burn in)", 
			ylab = "log proportional likelihood")
	if (num_trees > 1){
		for (t in 2 : nrow(all_tree_liks)){
			points(1 : (num_gibbs + 1), all_tree_liks[t, ], col = sample(COLORS, 1), pch = ".", cex = 2)
		}
	}
	abline(v = num_burn_in, col = "gray")
	if (save_plot){	
		dev.off()
	}	
}

hist_tree_liks = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	java_bart_machine = bart_machine$java_bart_machine
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	all_tree_liks = matrix(NA, nrow = num_trees, ncol = num_gibbs + 1)	
	for (t in 1 : num_trees){
		all_tree_liks[t, ] = .jcall(java_bart_machine, "[D", "getLikForTree", as.integer(t - 1))
	}	
	assign("all_tree_liks", all_tree_liks, .GlobalEnv)
	
	if (save_plot){
		save_plot_function(bart_machine, "tree_liks_hist", data_title)
	}	
	else {
		dev.new()
	}	
	
	all_liks_after_burn_in = as.vector(all_tree_liks[, (num_burn_in + 1) : (num_gibbs + 1)])
	min_lik = min(all_liks_after_burn_in)
	max_lik = max(all_liks_after_burn_in)	
	hist(all_liks_after_burn_in, 
			col = "white", 
			br = seq(from = min_lik - 0.01, to = max_lik + 0.01, by = (max_lik - min_lik + 0.01) / 100), 
			border = NA, 
			main = "Histogram of tree prop likelihoods after burn-in")
#	shapiro.test(all_liks_after_burn_in)
	for (t in 1 : num_trees){
		hist(all_tree_liks[t, (num_burn_in + 1) : (num_gibbs + 1)], 
				col = rgb(runif(1, 0, 0.8), runif(1, 0, 0.8), runif(1, 0, 0.8), ALPHA), 
				border = NA, 
				br = seq(from = min_lik, to = max_lik, by = (max_lik - min_lik) / 100), 
				add = TRUE)
		
	}
	
	if (save_plot){	
		dev.off()
	}	
}

plot_y_vs_yhat_a_bart = function(bart_predictions, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	y = bart_predictions[["y"]]
	n = length(y)
	y_hat = bart_predictions[["y_hat"]]
	ppi_a = bart_predictions[["ppi_a"]]
	ppi_b = bart_predictions[["ppi_b"]]
	ppi_conf = bart_predictions[["ppi_conf"]]
	y_inside_ppi = bart_predictions[["y_inside_ppi"]]
	prop_ys_in_ppi = bart_predictions[["prop_ys_in_ppi"]]
	L1_err = round(bart_predictions[["L1_err"]])
	L2_err = round(bart_predictions[["L2_err"]])
	rmse = bart_predictions[["rmse"]]
	
	#make the general plot
	if (save_plot){	
		save_plot_function(bart_machine, "yvyhat_A_Bart", data_title)
	}
	else {
		dev.new()
	}		
	plot(y, 
		y_hat, 
		main = paste("BART y-yhat, ", (ppi_conf * 100), "% PPIs (", round(prop_ys_in_ppi * 100, 2), "% cvrg), L1/2 = ", L1_err, "/", L2_err, ", rmse = ", round(rmse, 2), ifelse(is.null(extra_text), "", paste("\n", extra_text)), sep = ""), 
		xlab = paste("y\ngreen circle - the PPI for yhat captures true y", sep = ""), 
		ylab = "y_hat", 
		cex = 0)
	#draw PPI's
	for (i in 1 : n){
		segments(y[i], ppi_a[i], y[i], ppi_b[i], col = "black", lwd = 0.1)	
	}
	#draw green dots or red dots depending upon inclusion in the PPI
	for (i in 1 : n){
		points(y[i], y_hat[i], col = ifelse(y_inside_ppi[i], "green", "red"), cex = 0.3, pch = 16)	
	}
	abline(a = 0, b = 1, col = "blue")
	if (save_plot){	
		dev.off()
	}		
}

look_at_sample_of_test_data = function(bart_predictions, grid_len = 3, extra_text = NULL){
	par(mfrow = c(grid_len, grid_len))
	dev.new() 
	for (i in sample(1 : n, grid_len^2)){
		y_i = test_data$y[i]
		samps = as.numeric(y_hat_posterior_samples[i, ])
		hist(samps, 
			br = 50, 
			main = paste("point #", i, "in the dataset y =", round(y_i, 2), ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
			xlim = c(min(y_i, samps), max(y_i, samps)))
		abline(v = y_hat[i], col = "purple", lwd = 3)
		abline(v = sample_mode(samps), col = "blue")
		abline(v = median(samps), col = "red")
		abline(v = y_i, col = "green", lwd = 3)
		abline(v = ppi_b[i], col = "grey")
		abline(v = ppi_a[i], col = "grey")
	}	
}

bart_predict_for_test_data = function(bart_machine, test_data, num_cores = 1){
	y_hat = bart_predict(bart_machine, test_data, num_cores)
	y = test_data$y
	n = nrow(test_data)
	
	predict_obj$y_hat = y_hat
	predict_obj$L1_err = sum(abs(y - y_hat))
	predict_obj$L2_err = sum((y - y_hat)^2)
	predict_obj$rmse = sqrt(predict_obj$L2_err / n)
	predict_obj$e = y - y_hat
	
	predict_obj
}

bart_predict = function(bart_machine, new_data, num_cores = 1){
	#pull out data objects for convenience
	java_bart_machine = bart_machine$java_bart_machine
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	n = nrow(new_data)
	
	
	#check for errors in data
	if (error_in_data(new_data)){
		return;
	}	
	new_data = pre_process_data(new_data)	
	
	y_hat = array(NA, n)
	for (i in 1 : n){
		y_hat[i] = .jcall(java_bart_machine, "D", "Evaluate", c(as.numeric(new_data[i, ])), as.integer(num_cores))
	}
	
	y_hat
}

calc_ppis_from_prediction = function(bart_machine, new_data, ppi_conf = 0.95, num_cores = 1){
	n_test = nrow(new_data)
	
	ppi_lower_bd = array(NA, n_test)
	ppi_upper_bd = array(NA, n_test)	
	
	y_hat_posterior_samples = matrix(NA, nrow = n_test, ncol = bart_machine$num_iterations_after_burn_in)	
	for (i in 1 : n_test){
		y_hat_posterior_samples[i, ] = 
			.jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesForPrediction", c(as.numeric(new_data[i, ]), NA), as.integer(num_cores))
	}
	#to get y_hat.. just take straight mean of posterior samples, alternatively, we can let java do it if we want more bells and whistles
	y_hat = calc_y_hat_from_gibbs_samples(y_hat_posterior_samples)	
	
	for (i in 1 : bart_machine$n){		
		ppi_lower_bd[i] = quantile(sort(y_hat_posterior_samples[i, ]), (1 - ppi_conf) / 2)
		ppi_upper_bd[i] = quantile(sort(y_hat_posterior_samples[i, ]), (1 + ppi_conf) / 2)
	}	
	#did the PPI capture the true y?
	y_inside_ppi = new_data$y >= ppi_lower_bd & new_data$y <= ppi_upper_bd 
	
	list(ppis = cbind(ppi_lower_bd, ppi_upper_bd),
		ppi_conf = ppi_conf,
		y_inside_ppi = y_inside_ppi,
		prop_ys_in_ppi = sum(y_inside_ppi) / n_test
	)
}

#let's do sample medians like Rob
calc_y_hat_from_gibbs_samples = function(y_hat_posterior_samples){
	apply(y_hat_posterior_samples, 1, mean)
}

run_other_model_and_plot_y_vs_yhat = function(y_hat, 
		model_name, 
		test_data, 
		training_data,
		extra_text = NULL, 
		data_title = "data_model", 
		save_plot = FALSE,
		bart_machine = NULL, 
		sigsqs = NULL,
		avg_num_splits_by_vars = NULL,
		y_hat_train = NULL,
		runtime = NULL,
		create_plot = FALSE){
	L1_err = sum(abs(test_data$y - y_hat))
	L2_err = sum((test_data$y - y_hat)^2)	
	rmse = sqrt(L2_err / length(y_hat))
	L2_err_train = sum((training_data$y - y_hat_train)^2)
	rmse_train = sqrt(L2_err_train / length(y_hat_train))
	
	if (create_plot){
		if (save_plot){
			save_plot_function(bart_machine, paste("yvyhat_", model_name, sep = ""), data_title)
		}	
		else {
			dev.new()
		}		
		plot(test_data$y, 
				y_hat, 
				main = paste("y/yhat ", model_name, " model L1/2 = ", round(L1_err, 1), "/", round(L2_err, 1), " rmse = ", round(rmse, 2), ifelse(is.null(extra_text), "", paste("\n", extra_text)), sep = ""), 
				xlab = "y", 
				ylab = "y_hat")	
		if (save_plot){	
			dev.off()
		}
	}
	
	list(y_hat = y_hat, 
		L1_err = L1_err, 
		L2_err = L2_err, 
		rmse = rmse, 
		sigsqs = sigsqs,
		avg_num_splits_by_vars = avg_num_splits_by_vars,
		rmse_train = rmse_train,
		runtime = runtime)		
}

get_variable_significance = function(bart_machine, var_num, data = NULL, num_iter = 100, print_histogram = TRUE, num_cores = 1){
	if (bart_machine$run_in_sample){
		real_sse = bart_machine$L2_err_train
		n = bart_machine$n
		data = bart_machine$training_data
	} else {
		real_sse = bart_predict_for_test_data(bart_machine, data, num_cores)$L2_err
		n = nrow(data)
	}
  
  
  sse_vec = array(NA, num_iter)
  
  for (i in 1 : num_iter){
    data_scrambled = data
    data_scrambled[, var_num] = data[sample(1 : n, n, replace = FALSE), var_num] ##scrambled column of interest
    sse_vec[i] = bart_predict_for_test_data(bart_machine, data_scrambled, num_cores)$L2_err
    if (i %% 10 == 0) print(i)
  }
  p_value = (1 + sum(real_sse > sse_vec)) / (1 + num_iter) ##how many null values greater than obs. 
  if (print_histogram){
	windows()
    hist(sse_vec, 
		 br = num_iter / 3,
         col = "grey", 
         main = paste("Null Distribution for Variable", var_num),
         xlim = c(min(sse_vec,real_sse-1),max(sse_vec, real_sse+1)),
         xlab = "SSE")
    abline(v=real_sse, col="blue", lwd = 3)
  }
  list(p_value = p_value, sse_vec = sse_vec, real_sse = real_sse)
}


run_random_forests_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	before = Sys.time()
	rf_mod = randomForest(y ~., training_data)
	y_hat = predict(rf_mod, test_data)
	after = Sys.time()
	print(paste("RF run time:", after - before))	
	run_other_model_and_plot_y_vs_yhat(y_hat, 
		"RF", 
		test_data, 
		training_data, 
		extra_text, 
		data_title, 
		save_plot, 
		bart_machine,
		runtime = after - before)
}

run_bayes_tree_bart_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	p = ncol(training_data) - 1
	before = Sys.time()
	bayes_tree_bart_mod = bart(x.train = training_data[, 1 : p],
		y.train = training_data$y,
		sigest = sd(training_data$y), #we force the sigma estimate to be the std dev of y
		x.test = test_data[, 1 : p],
		sigdf = 3, #$\nu = 3$, this is the same value we used in the implementation
		sigquant = 0.9, 
		k = 2, #same as we have it
		power = DEFAULT_BETA, #same as we have it
		base = DEFAULT_ALPHA, #same as we have it
		ntree = bart_machine$num_trees, #keep it the same -- default is 200 in BayesTree... interesting...
		ndpost = bart_machine$num_iterations_after_burn_in, #keep it the same
		nskip = bart_machine$num_burn_in, #keep it the same --- default is 100 in BayesTree -- huh??
		usequants = TRUE, #this is a tiny bit different...check with Ed
		numcut = length(training_data$y), #this is a tiny bit different...check with Ed
		verbose = TRUE)
		
#	out = list(yhat = bayes_tree_bart_mod$yhat.test.mean, sigmas = bayes_tree_bart_mod$sigma)
	y_hat = bayes_tree_bart_mod$yhat.test.mean
	after = Sys.time()
	print(paste("R BART run time:", after - before))	
	sigsqs = (bayes_tree_bart_mod$sigma)^2
	avg_num_splits_by_vars = tryCatch({apply(bayes_tree_bart_mod$varcount, 2, mean)}, error = function(e){NA})
	y_hat_train = bayes_tree_bart_mod$yhat.train.mean
	run_other_model_and_plot_y_vs_yhat(y_hat, 
		"R_BART", 
		test_data, 
		training_data,
		extra_text, 
		data_title, 
		save_plot, 
		bart_machine, 
		sigsqs = sigsqs, 
		avg_num_splits_by_vars = avg_num_splits_by_vars,
		y_hat_train = y_hat_train,
		runtime = after - before)
}

run_cart_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	before = Sys.time()
	cart_model = rpart(y ~ ., training_data)
	y_hat = predict(cart_model, test_data)
	after = Sys.time()
	print(paste("CART run time:", after - before))		
	run_other_model_and_plot_y_vs_yhat(y_hat, 
		"CART", 
		test_data, 
		training_data, 
		extra_text, 
		data_title, 
		save_plot, 
		bart_machine,
		runtime = after - before)
}

error_in_data = function(data){
	if (is.null(colnames(data))){
		stop("no colnames in data matrix", call. = FALSE)
		return(TRUE)
	}
	if (colnames(data)[ncol(data)] != "y"){
		stop("last column of BART data must be the response and it must be named \"y\"", call. = FALSE)
		return(TRUE)
	}
	FALSE
}

pre_process_data = function(data){
	#delete missing data just in case
	data = data[!is.na(data), ]
	#convert to model matrix with binary dummies (if factor has more than k>2 levels, we need dummies for all k, not k-1 (thanks Andreas) 
	#see http://stackoverflow.com/questions/4560459/all-levels-of-a-factor-in-a-model-matrix-in-r
	model_matrix = model.matrix(~ ., data)
	#kill intercept since it's useless for BART anyway
	model_matrix[, 2 : ncol(model_matrix)]
}

#believe it or not... there's no standard R function for this, isn't that pathetic?
sample_mode = function(data){
	as.numeric(names(sort(-table(data)))[1])
}

save_plot_function = function(bart_machine, identifying_text, data_title){
	if (is.null(bart_machine)){
		stop("you cannot save a plot unless you pass the bart_machine object", call. = FALSE)
	}
	num_iterations_after_burn_in = bart_machine[["num_iterations_after_burn_in"]]
	num_burn_in = bart_machine$num_burn_in
	num_trees = bart_machine$num_trees	
	alpha = bart_machine$alpha
	beta = bart_machine$beta
	plot_filename = paste(PLOTS_DIR, "/", data_title, "_", identifying_text, "_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, "_alpha_", alpha, "_beta_", beta, ".pdf", sep = "")
	tryCatch({pdf(file = plot_filename)}, error = function(e){})
	append_to_log(paste("plot saved as", plot_filename))
}