
#libraries and dependencies
tryCatch(library(randomForest), error = function(e){install.packages("randomForest")}, finally = library(randomForest))
tryCatch(library(rJava), error = function(e){install.packages("rJava")}, finally = library(rJava))

#constants
NUM_GIGS_RAM_TO_USE = ifelse(.Platform$OS.type == "windows", 4, 8)
PLOTS_DIR = "output_plots"
JAR_DEPENDENCIES = c("bart_java.jar", "commons-math-2.1.jar", "jai_codec.jar", "jai_core.jar", "gemident_1_12_12.jar")
DATA_FILENAME = "datasets/bart_data.csv"

bart_model = function(training_data, num_trees = 50, num_burn_in = 1000, num_iterations_after_burn_in = 1000, class_or_regr = "r", debug_log = FALSE){
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
	bart_machine = init_jvm_and_bart_object(debug_log)
	#make bart to spec with what the user wants
	.jcall(bart_machine, "V", "setNumTrees", as.integer(num_trees))
	.jcall(bart_machine, "V", "setNumGibbsBurnIn", as.integer(num_burn_in))
	.jcall(bart_machine, "V", "setNumGibbsTotalIterations", as.integer(num_gibbs))
	
	#build the bart machine for use later
	#need http://math.acadiau.ca/ACMMaC/Rmpi/sample.html RMPI here
	#or better yet do this in pieces and plot slowly
	.jcall(bart_machine, "V", "Build")
	
	#now once it's done, let's extract things that are related to diagnosing the build of the BART model
	sigsqs = .jcall(bart_machine, "[D", "getGibbsSamplesSigsqs")
	
	#need to matrix this up
	all_tree_liks = matrix(NA, nrow = num_trees, ncol = num_gibbs + 1)
	for (t in 1 : num_trees){
		all_tree_liks[t, ] = .jcall(bart_machine, "[D", "getLikForTree", as.integer(t - 1))
	}
	
	
#	while (TRUE){
#		if (.jcall(bart_machine, "Z", "gibbsFinished")){
#			break
#		}
#		cat(paste("BART iteration:", .jcall(bart_machine, "I", "currentGibbsSampleIteration")))
#	}
	list(bart_machine = bart_machine, 
		num_trees = num_trees,
		num_burn_in = num_burn_in,
		num_iterations_after_burn_in = num_iterations_after_burn_in, 
		num_gibbs = num_gibbs,
		sigsqs = sigsqs,
		all_tree_liks = all_tree_liks)
}

init_jvm_and_bart_object = function(debug_log){
	.jinit(parameters = paste("-Xmx", NUM_GIGS_RAM_TO_USE, "g", sep = ""))
	for (dependency in JAR_DEPENDENCIES){
		.jaddClassPath(paste(directory_where_code_is, "/", dependency, sep = ""))
	}
	bart_machine = .jnew("CGM_BART.CGMBARTRegression")
	if (debug_log){
		.jcall(bart_machine, "V", "writeToDebugLog")	
	}
	bart_machine
}

plot_sigsqs_convergence_diagnostics = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	sigsqs = bart_machine[["sigsqs"]]
	num_iterations_after_burn_in = bart_machine[["num_iterations_after_burn_in"]]
	num_burn_in = bart_machine[["num_burn_in"]]
	num_gibbs = bart_machine[["num_gibbs"]]
	num_trees = bart_machine[["num_trees"]]
	
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

plot_tree_liks_convergence_diagnostics = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	all_tree_liks = bart_machine[["all_tree_liks"]]
	num_burn_in = bart_machine[["num_burn_in"]]
	num_gibbs = bart_machine[["num_gibbs"]]
	
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
	for (t in 2 : nrow(all_tree_liks)){
		tryCatch({points(1 : (num_gibbs + 1), all_tree_liks[t, ], col = sample(colors(), 1), main = "tree log proportional likelihood", pch = ".", cex = 2)},
			error = function(exc){},
			finally = function(exc){})
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
	L1_err = bart_predictions[["L1_err"]]
	L2_err = bart_predictions[["L2_err"]]

	#make the general plot
	if (save_plot){	
		save_plot_function(bart_machine, "yvyhat_bart", data_title)
	}
	else {
		dev.new()
	}		
	plot(y, 
		y_hat, 
		main = paste("BART y vs yhat, ", (ppi_conf * 100), "% PPIs (", round(prop_ys_in_ppi * 100, 2), "% cvrg), L1err = ", L1_err, ", L2err = ", L2_err, ifelse(is.null(extra_text), "", paste("\n", extra_text)), sep = ""), 
		xlab = paste("y\ngreen circle - the PPI for yhat captures true y", sep = ""), 
		ylab = "y_hat", 
		cex = 0)
	#draw PPI's
	for (i in 1 : n){
		segments(y[i], ppi_a[i], y[i], ppi_b[i], col = "black", lwd = 0.1)	
	}
	#draw green dots or red dots depending upon inclusion in the PPI
	for (i in 1 : n){
		points(y[i], y_hat[i], col = ifelse(y_inside_ppi[i], "green", "red"), cex = 0.1, pch = 16)	
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


predict_and_calc_ppis = function(model, test_data, ppi_conf = 0.95){
	#pull out data objects for convenience
	bart_machine = model[["bart_machine"]]
	num_iterations_after_burn_in = model[["num_iterations_after_burn_in"]]
	n = nrow(test_data)
	y = test_data$y

	#check for errors in data
	if (error_in_data(test_data)){
		return;
	}	
	test_data = pre_process_data(test_data)
	if (!.jcall(bart_machine, "Z", "gibbsFinished")){
		stop("BART model not finished building yet", call. = FALSE)
		return
	}
	
	#do the evaluation as a loop... one day this should be done better than this with a matrix
	y_hat_posterior_samples = matrix(NA, nrow = nrow(test_data), ncol = num_iterations_after_burn_in) 
	ppi_a = array(NA, n)
	ppi_b = array(NA, n)	
	for (i in 1 : n){
		samps = .jcall(bart_machine, "[D", "getGibbsSamplesForPrediction", c(as.numeric(test_data[i, ]), NA))
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
		L2_err = L2_err)
}

run_random_forests_and_plot_y_vs_yhat = function(training_data, test_data, extra_text = NULL, data_title = "data_model", save_plot = FALSE, bart_machine = NULL){
	rf_mod = randomForest(y ~., training_data)
	yhat_rf = predict(rf_mod, test_data)
	
	l1err_rf = round(sum(abs(test_data$y - yhat_rf)), 1)
	l2err_rf = round(sum((test_data$y - yhat_rf)^2), 1)	
	
	if (save_plot){
		save_plot_function(bart_machine, "yvyhat_rf", data_title)
	}	
	else {
		dev.new()
	}		
	plot(test_data$y, 
			yhat_rf, 
			main = paste("y vs yhat Random Forests model L1err = ", l1err_rf, " L2err = ", l2err_rf, ifelse(is.null(extra_text), "", paste("\n", extra_text)), sep = ""), 
			xlab = "y", 
			ylab = "y_hat")	
	dev.off()

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