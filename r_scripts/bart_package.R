
#libraries and dependencies
options(repos = "http://lib.stat.cmu.edu/R/CRAN")
tryCatch(library(randomForest), error = function(e){install.packages("randomForest")}, finally = library(randomForest))
tryCatch(library(rJava), error = function(e){install.packages("rJava")}, finally = library(rJava))
tryCatch(library(rpart), error = function(e){install.packages("rpart")}, finally = library(rpart))
tryCatch(library(xtable), error = function(e){install.packages("xtable")}, finally = library(xtable))

if (.Platform$OS.type == "windows"){
	tryCatch(library(BayesTree), error = function(e){install.packages("BayesTree")}, finally = library(BayesTree))
} else {
	library(BayesTree, lib.loc = "~/R/")
}


#constants
NUM_GIGS_RAM_TO_USE = ifelse(.Platform$OS.type == "windows", 6, 8)
PLOTS_DIR = "output_plots"
JAR_DEPENDENCIES = c("bart_java.jar", "commons-math-2.1.jar", "jai_codec.jar", "jai_core.jar")
DATA_FILENAME = "datasets/bart_data.csv"
DEFAULT_ALPHA = 0.95
DEFAULT_BETA = 2
COLORS = array(NA, 500)
for (i in 1 : 500){
	COLORS[i] = rgb(runif(1, 0, 0.7), runif(1, 0, 0.7), runif(1, 0, 0.7))
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
JAVA_LOG = TRUE #to be overwritten later
#source("r_scripts/create_simulated_models.R")
#simulated_data_model_name = simulated_data_sets[1]
#training_data = simulate_data_from_simulation_name(simulated_data_model_name)
#test_data = simulate_data_from_simulation_name(simulated_data_model_name)

bart_model = function(training_data, 
		num_trees = 50, 
		num_burn_in = 1000, 
		num_iterations_after_burn_in = 1000, 
		alpha = DEFAULT_ALPHA,
		beta = DEFAULT_BETA, 
		class_or_regr = "r", 
		debug_log = TRUE,
		fix_seed = FALSE,
		unique_name = "unnamed",
		print_tree_illustrations = FALSE, 
		print_out_every = NULL){
	
#	clean_previous_bart_data()

	num_gibbs = num_burn_in + num_iterations_after_burn_in
	#check for errors in data
	if (error_in_data(training_data)){
		return;
	}
	training_data = pre_process_data(training_data)
	
	#now write the data as a csv, make sure to exclude row names,
	#this is how we're going to send the data in BART for now even
	#though it is inefficient
	write.csv(training_data, DATA_FILENAME, row.names = FALSE)
	
	#initialize the JVM
	java_bart_machine = init_jvm_and_bart_object(unique_name, debug_log, print_tree_illustrations, print_out_every, fix_seed)
	#make bart to spec with what the user wants
	.jcall(java_bart_machine, "V", "setNumTrees", as.integer(num_trees))
	.jcall(java_bart_machine, "V", "setNumGibbsBurnIn", as.integer(num_burn_in))
	.jcall(java_bart_machine, "V", "setNumGibbsTotalIterations", as.integer(num_gibbs))
	.jcall(java_bart_machine, "V", "setAlpha", alpha)
	.jcall(java_bart_machine, "V", "setBeta", beta)
	
	
	
	#build the bart machine for use later
	#need http://math.acadiau.ca/ACMMaC/Rmpi/sample.html RMPI here
	#or better yet do this in pieces and plot slowly
	.jcall(java_bart_machine, "V", "Build")

	#now once it's done, let's extract things that are related to diagnosing the build of the BART model
	list(java_bart_machine = java_bart_machine, 
		num_trees = num_trees,
		num_burn_in = num_burn_in,
		num_iterations_after_burn_in = num_iterations_after_burn_in, 
		num_gibbs = num_gibbs,
		alpha = alpha,
		beta = beta
	)
}

printed_out_warnings = FALSE

init_jvm_and_bart_object = function(unique_name, debug_log, print_tree_illustrations, print_out_every, fix_seed){
	.jinit(parameters = paste("-Xmx", NUM_GIGS_RAM_TO_USE, "g", sep = ""))
	for (dependency in JAR_DEPENDENCIES){
		.jaddClassPath(paste(directory_where_code_is, "/", dependency, sep = ""))
	}
	java_bart_machine = .jnew("CGM_BART.CGMBARTRegression")
	
	#first set the name
	.jcall(java_bart_machine, "V", "setUniqueName", unique_name)
	#now set whether we want the program to log to a file
	if (debug_log){
		if (!printed_out_warnings){
			cat("warning: printing out the log file will slow down the runtime perceptibly\n")
		}
		.jcall(java_bart_machine, "V", "writeToDebugLog")
	}
	#now initialize the data
	.jcall(java_bart_machine, "V", "setDataToDefaultForRPackage")
	if (fix_seed){
		.jcall(java_bart_machine, "V", "fixRandSeed")		
	}
	#set whether we want there to be tree illustrations
	if (print_tree_illustrations){
		if (!printed_out_warnings){
			cat("warning: printing tree illustrations will slow down the runtime significantly\n")
		}
		.jcall(java_bart_machine, "V", "printTreeIllustations")
	}

	if (!is.null(print_out_every)){
		.jcall(java_bart_machine, "V", "setPrintOutEveryNIter", as.integer(print_out_every))
	}
	printed_out_warnings = TRUE
	java_bart_machine
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

#ensure_bart_is_done_in_java = function(java_bart_machine){
#	if (!.jcall(java_bart_machine, "Z", "gibbsFinished")){
#		stop("BART model not finished building yet", call. = FALSE)
#	}
#}


plot_sigsqs_convergence_diagnostics = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
	assign("sigsqs", sigsqs, .GlobalEnv)
	
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
}

hist_sigsqs = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
	assign("sigsqs", sigsqs, .GlobalEnv)
	
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

plot_y_vs_yhat = function(bart_predictions, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
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

predict_and_calc_ppis = function(bart_machine, test_data, ppi_conf = 0.95){
	#pull out data objects for convenience
	java_bart_machine = bart_machine$java_bart_machine
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	n = nrow(test_data)
	y = test_data$y

	#check for errors in data
	if (error_in_data(test_data)){
		return;
	}	
	test_data = pre_process_data(test_data)

	
	#do the evaluation as a loop... one day this should be done better than this with a matrix
	y_hat_posterior_samples = matrix(NA, nrow = nrow(test_data), ncol = num_iterations_after_burn_in) 
	ppi_a = array(NA, n)
	ppi_b = array(NA, n)	
	for (i in 1 : n){
#		tryCatch({
		samps = .jcall(java_bart_machine, "[D", "getGibbsSamplesForPrediction", c(as.numeric(test_data[i, ]), NA))
#		},
#		error = function(exc){return},
#		finally = function(exc){})		
		y_hat_posterior_samples[i, ] = samps

		ppi_a[i] = quantile(sort(samps), (1 - ppi_conf) / 2)
		ppi_b[i] = quantile(sort(samps), (1 + ppi_conf) / 2)
	}
	#to get y_hat.. just take straight mean of posterior samples, alternatively, we can let java do it if we want more bells and whistles
	y_hat = calc_y_hat_from_gibbs_samples(y_hat_posterior_samples)
	assign("y_hat", y_hat, .GlobalEnv)
	#did the PPI capture the true y?
	y_inside_ppi = y >= ppi_a & y <= ppi_b
	prop_ys_in_ppi = sum(y_inside_ppi) / n
	
	L1_err = round(sum(abs(y - y_hat)), 1)
	L2_err = round(sum((y - y_hat)^2), 1)		
	
	#give the user everything
	list(y_hat_posterior_samples = y_hat_posterior_samples, 
		y = test_data$y,
		y_hat = y_hat, 
		ppi_a = ppi_a, 
		ppi_b = ppi_b, 
		ppi_conf = ppi_conf,
		y_inside_ppi = y_inside_ppi, 
		prop_ys_in_ppi = prop_ys_in_ppi,
		L1_err = L1_err,
		L2_err = L2_err,
		rmse = sqrt(L2_err / n))
}

#let's do sample medians like Rob
calc_y_hat_from_gibbs_samples = function(y_hat_posterior_samples){
	apply(y_hat_posterior_samples, 1, function(row){median(row)})
}

run_other_model_and_plot_y_vs_yhat = function(y_hat, model_name, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	L1_err = round(sum(abs(test_data$y - y_hat)), 1)
	L2_err = round(sum((test_data$y - y_hat)^2), 1)	
	rmse = sqrt(L2_err / length(y_hat))
	
	if (save_plot){
		save_plot_function(bart_machine, paste("yvyhat_", model_name, sep = ""), data_title)
	}	
	else {
		dev.new()
	}		
	plot(test_data$y, 
			y_hat, 
			main = paste("y/yhat ", model_name, " model L1/2 = ", L1_err, "/", L2_err, " rmse = ", round(rmse, 2), ifelse(is.null(extra_text), "", paste("\n", extra_text)), sep = ""), 
			xlab = "y", 
			ylab = "y_hat")	
	if (save_plot){	
		dev.off()
	}
	
	list(y_hat = y_hat, L1_err = L1_err, L2_err = L2_err, rmse = rmse)		
}

run_random_forests_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	rf_mod = randomForest(y ~., training_data)
	y_hat = predict(rf_mod, test_data)
	run_other_model_and_plot_y_vs_yhat(y_hat, "RF", test_data, extra_text, data_title, save_plot, bart_machine)
}

run_bayes_tree_bart_impl_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	p = ncol(training_data) - 1
	y = training_data[, p + 1]
	bayes_tree_bart_mod = bart(x.train = training_data[, 1 : p],
		y.train = y,
		sigest = sd(y), #we force the sigma estimate to be the std dev of y
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
		numcut = length(y), #this is a tiny bit different...check with Ed
		verbose = FALSE)
		
	y_hat = bayes_tree_bart_mod$yhat.test.mean
	run_other_model_and_plot_y_vs_yhat(y_hat, "R_BART", test_data, extra_text, data_title, save_plot, bart_machine)
}

run_cart_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	cart_model = rpart(y ~ ., training_data)
	y_hat = predict(cart_model, test_data)
	run_other_model_and_plot_y_vs_yhat(y_hat, "CART", test_data, extra_text, data_title, save_plot, bart_machine)
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
	#TO-DO
	#MUST BE DETERMINISTIC!
	#1) convert categorical data into 0/1's
	#2) delete missing data rows
	data
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
	pdf(file = plot_filename)
	append_to_log(paste("plot saved as", plot_filename))
}