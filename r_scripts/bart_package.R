
#libraries and dependencies
tryCatch(library(randomForest), error = function(e){install.packages("randomForest")}, finally = library(randomForest))
tryCatch(library(rJava), error = function(e){install.packages("rJava")}, finally = library(rJava))
tryCatch(library(BayesTree), error = function(e){install.packages("BayesTree")}, finally = library(BayesTree))
tryCatch(library(rpart), error = function(e){install.packages("rpart")}, finally = library(rpart))

#constants
NUM_GIGS_RAM_TO_USE = ifelse(.Platform$OS.type == "windows", 4, 8)
PLOTS_DIR = "output_plots"
JAR_DEPENDENCIES = c("bart_java.jar", "commons-math-2.1.jar", "jai_codec.jar", "jai_core.jar", "gemident_1_12_12.jar")
DATA_FILENAME = "datasets/bart_data.csv"

#some defaults if you want to run this
#num_trees = 5
#num_burn_in = 100
#num_iterations_after_burn_in = 100
#num_gibbs = num_burn_in + num_iterations_after_burn_in
#
#class_or_regr = "r"
#debug_log = TRUE
#print_tree_illustrations = FALSE
#print_out_every = NULL
#source("r_scripts/create_simulated_models.R")
#simulated_data_model_name = simulated_data_model_names[1]
#training_data = simulate_data_from_simulation_name(simulated_data_model_name)
#test_data = simulate_data_from_simulation_name(simulated_data_model_name)

bart_model = function(training_data, 
		num_trees = 50, 
		num_burn_in = 1000, 
		num_iterations_after_burn_in = 1000, 
		class_or_regr = "r", 
		debug_log = FALSE, 
		print_tree_illustrations = FALSE, 
		print_out_every = NULL){

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
	java_bart_machine = init_jvm_and_bart_object(debug_log, print_tree_illustrations, print_out_every)
	#make bart to spec with what the user wants
	.jcall(java_bart_machine, "V", "setNumTrees", as.integer(num_trees))
	.jcall(java_bart_machine, "V", "setNumGibbsBurnIn", as.integer(num_burn_in))
	.jcall(java_bart_machine, "V", "setNumGibbsTotalIterations", as.integer(num_gibbs))
	
	#build the bart machine for use later
	#need http://math.acadiau.ca/ACMMaC/Rmpi/sample.html RMPI here
	#or better yet do this in pieces and plot slowly
	.jcall(java_bart_machine, "V", "Build")
#	print(java_bart_machine)
	#now once it's done, let's extract things that are related to diagnosing the build of the BART model
	ensure_bart_is_done_in_java(java_bart_machine)
	sigsqs = .jcall(java_bart_machine, "[D", "getGibbsSamplesSigsqs")
	
	list(java_bart_machine = java_bart_machine, 
		num_trees = num_trees,
		num_burn_in = num_burn_in,
		num_iterations_after_burn_in = num_iterations_after_burn_in, 
		num_gibbs = num_gibbs)
}

############
#for (i in 1 : 100){
#	.jcall(java_bart_machine, "V", "Build")
#	print(java_bart_machine)
#	#now once it's done, let's extract things that are related to diagnosing the build of the BART model
#	ensure_bart_is_done_in_java(java_bart_machine)
#	sigsqs = .jcall(java_bart_machine, "[D", "getGibbsSamplesSigsqs")	
#}

ensure_bart_is_done_in_java = function(java_bart_machine){
	if (!.jcall(java_bart_machine, "Z", "gibbsFinished")){
		stop("BART model not finished building yet", call. = FALSE)
	}	
}

init_jvm_and_bart_object = function(debug_log, print_tree_illustrations, print_out_every){
	.jinit(parameters = paste("-Xmx", NUM_GIGS_RAM_TO_USE, "g", sep = ""))
	for (dependency in JAR_DEPENDENCIES){
		.jaddClassPath(paste(directory_where_code_is, "/", dependency, sep = ""))
	}
	java_bart_machine = .jnew("CGM_BART.CGMBARTRegression")
	if (debug_log){
#		cat("warning: printing out the log file will slow down the runtime perceptibly\n")
		.jcall(java_bart_machine, "V", "writeToDebugLog")
	}
	if (print_tree_illustrations){
		cat("warning: printing tree illustrations will slow down the runtime significantly\n")
		.jcall(java_bart_machine, "V", "printTreeIllustations")
	}
	if (!is.null(print_out_every)){
		.jcall(java_bart_machine, "V", "setPrintOutEveryNIter", as.integer(print_out_every))
	}
	java_bart_machine
}

plot_sigsqs_convergence_diagnostics = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
#	tryCatch({
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
#	},
#	error = function(exc){return},
#	finally = function(exc){})
	
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	#first look at sigsqs
	avg_sigsqs = mean(sigsqs[(length(sigsqs) - num_iterations_after_burn_in) : length(sigsqs)], na.rm = TRUE)
	 
	if (save_plot){
		save_plot_function(bart_machine, "sigsqs", data_title)
	}	
	else {
		dev.new()
	}
	plot(sigsqs, 
			main = paste("Sigsq throughout entire project  sigsq after burn in ~ ", round(avg_sigsqs, 2), ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
			xlab = "Gibbs sample # (gray line indicates burn-in, yellow lines are the 95% PPI after burn-in)", 
			ylim = c(max(min(sigsqs) - 0.1, 0), min(sigsqs) + 7.5 * min(sigsqs)),
			pch = ".", 
			cex = 3,
			col = "gray")
	points(sigsqs, pch = ".", col = "red")
	ppi_sigsqs = quantile(sigsqs[(length(sigsqs) - num_gibbs) : length(sigsqs)], c(.025, .975))
	abline(a = ppi_sigsqs[1], b = 0, col = "yellow")
	abline(a = ppi_sigsqs[2], b = 0, col = "yellow")	
	abline(v = num_burn_in, col = "gray")
	
	dev.off()
	
}

plot_tree_num_nodes = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	java_bart_machine = bart_machine$java_bart_machine
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	all_tree_num_nodes = matrix(NA, nrow = num_trees, ncol = num_gibbs + 1)
	for (n in 1 : (num_gibbs + 1)){
#		tryCatch({
		all_tree_num_nodes[, n] = .jcall(java_bart_machine, "[I", "getNumNodesForTreesInGibbsSamp", as.integer(n - 1))
#		},
#		error = function(exc){return},
#		finally = function(exc){})
	}
	
	if (save_plot){
		save_plot_function(bart_machine, "tree_nodes", data_title)
	}	
	else {
		dev.new()
	}	
	plot(1 : (num_gibbs + 1),  # + 1 for the prior
			all_tree_num_nodes[1, ], 
			col = sample(colors(), 1),
			pch = ".",
			type = "l", 
			main = paste("Num Nodes by Iteration", ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
			ylim = c(min(all_tree_num_nodes), max(all_tree_num_nodes)),
			xlab = "Gibbs sample # (gray line indicates burn in)", 
			ylab = "num nodes")
	if (num_trees > 1){
		for (t in 2 : nrow(all_tree_num_nodes)){
			points(1 : (num_gibbs + 1), all_tree_num_nodes[t, ], col = sample(colors(), 1), pch = ".", type = "l", cex = 2)
		}
	}
	abline(v = num_burn_in, col = "gray")
	
	dev.off()
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
#		tryCatch({
		all_tree_depths[, n] = .jcall(java_bart_machine, "[I", "getDepthsForTreesInGibbsSamp", as.integer(n - 1))
#		},
#		error = function(exc){return},
#		finally = function(exc){})		
		
	}
	
	if (save_plot){
		save_plot_function(bart_machine, "tree_depths", data_title)
	}	
	else {
		dev.new()
	}	
	plot(1 : (num_gibbs + 1),  # + 1 for the prior
			all_tree_depths[1, ], 
			col = sample(colors(), 1),
			pch = ".",
			type = "l", 
			main = paste("Depth by Iteration", ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
			ylim = c(min(all_tree_depths), max(all_tree_depths)),
			xlab = "Gibbs sample # (gray line indicates burn in)", 
			ylab = "depth")
	if (num_trees > 1){
		for (t in 2 : nrow(all_tree_depths)){
			points(1 : (num_gibbs + 1), all_tree_depths[t, ], col = sample(colors(), 1), pch = ".", type = "l", cex = 2)
		}
	}
	abline(v = num_burn_in, col = "gray")
	
	dev.off()
}

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
	
	treeliks_scale = (max(all_tree_liks) - min(all_tree_liks)) * 0.05
	
	if (save_plot){
		save_plot_function(bart_machine, "tree_liks", data_title)
	}	
	else {
		dev.new()
	}	
	plot(1 : (num_gibbs + 1),  # + 1 for the prior
			all_tree_liks[1, ], 
			col = sample(colors(), 1),
			pch = ".",
			main = paste("Tree ln(prop Lik) by Iteration", ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
			ylim = c(max(all_tree_liks) - treeliks_scale, max(all_tree_liks) + 200),
			xlab = "Gibbs sample # (gray line indicates burn in)", 
			ylab = "log proportional likelihood")
	if (num_trees > 1){
		for (t in 2 : nrow(all_tree_liks)){
			points(1 : (num_gibbs + 1), all_tree_liks[t, ], col = sample(colors(), 1), pch = ".", cex = 2)
		}
	}
	abline(v = num_burn_in, col = "gray")
	
	dev.off()
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
		save_plot_function(bart_machine, "yvyhat_bart", data_title)
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

	dev.off()
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
		######## NOT RIGHT!!!
		ppi_a[i] = quantile(sort(samps), (1 - ppi_conf) / 2)
		ppi_b[i] = quantile(sort(samps), (1 + ppi_conf) / 2)
	}
	#to get y_hat.. just take straight mean of posterior samples, alternatively, we can let java do it if we want more bells and whistles
	y_hat = rowSums(y_hat_posterior_samples) / ncol(y_hat_posterior_samples)
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
	dev.off()
	
	list(y_hat = y_hat, L1_err = L1_err, L2_err = L2_err, rmse = rmse)		
}

run_random_forests_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	rf_mod = randomForest(y ~., training_data)
	y_hat = predict(rf_mod, test_data)
	run_other_model_and_plot_y_vs_yhat(y_hat, "RF", test_data, extra_text, data_title, save_plot, bart_machine)
}

run_bayes_tree_bart_impl_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	p = ncol(training_data) - 1
	bayes_tree_bart_mod = bart(x.train = training_data[, 1 : p],
		y.train = training_data[, p + 1],
		x.test = test_data[, 1 : p],
		sigdf = 3, #$\nu = 3$, this is the same value we used in the implementation
		ntree = bart_machine$num_trees, #keep it the same
		ndpost = bart_machine$num_iterations_after_burn_in,
		nskip = bart_machine$num_burn_in, #keep it the same --- default is 100 in BayesTree -- huh??
		usequants = TRUE,
		verbose = FALSE) #this is how I implemented BART - basically allowing any split point in the data itself
		
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
	num_burn_in = bart_machine[["num_burn_in"]]
	num_trees = bart_machine[["num_trees"]]	
	plot_filename = paste(PLOTS_DIR, "/", data_title, "_", identifying_text, "_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, ".pdf", sep = "")
	pdf(file = plot_filename)
	cat(paste("plot saved as", plot_filename, "\n"))
}