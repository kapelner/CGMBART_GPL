
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
	all_mu_vals_for_all_trees
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



#plot_tree_liks_convergence_diagnostics = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
#	java_bart_machine = bart_machine$java_bart_machine
#	num_burn_in = bart_machine$num_burn_in
#	num_gibbs = bart_machine$num_gibbs
#	num_trees = bart_machine$num_trees
#	
#	all_tree_liks = matrix(NA, nrow = num_trees, ncol = num_gibbs + 1)	
#	for (t in 1 : num_trees){		
##		tryCatch({
#		all_tree_liks[t, ] = .jcall(java_bart_machine, "[D", "getLikForTree", as.integer(t - 1))
##		},
##		error = function(exc){return},
##		finally = function(exc){})			
#	}	
#	assign("all_tree_liks", all_tree_liks, .GlobalEnv)
#	
#	treeliks_scale = (max(all_tree_liks, na.rm = TRUE) - min(all_tree_liks, na.rm = TRUE)) * 0.5
#	
#	if (save_plot){
#		save_plot_function(bart_machine, "tree_liks_by_gibbs", data_title)
#	}	
#	else {
#		dev.new()
#	}	
#	plot(1 : (num_gibbs + 1),  # + 1 for the prior
#			all_tree_liks[1, ], 
#			col = sample(COLORS, 1),
#			pch = ".",
#			main = paste("Tree ln(prop Lik) by Iteration", ifelse(is.null(extra_text), "", paste("\n", extra_text))), 
#			ylim = quantile(all_tree_liks[1, ], c(0, .999), na.rm = TRUE),
#			xlab = "Gibbs sample # (gray line indicates burn in)", 
#			ylab = "log proportional likelihood")
#	if (num_trees > 1){
#		for (t in 2 : nrow(all_tree_liks)){
#			points(1 : (num_gibbs + 1), all_tree_liks[t, ], col = sample(COLORS, 1), pch = ".", cex = 2)
#		}
#	}
#	abline(v = num_burn_in, col = "gray")
#	if (save_plot){	
#		dev.off()
#	}	
#}
#
#hist_tree_liks = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
#	java_bart_machine = bart_machine$java_bart_machine
#	num_burn_in = bart_machine$num_burn_in
#	num_gibbs = bart_machine$num_gibbs
#	num_trees = bart_machine$num_trees
#	
#	all_tree_liks = matrix(NA, nrow = num_trees, ncol = num_gibbs + 1)	
#	for (t in 1 : num_trees){
#		all_tree_liks[t, ] = .jcall(java_bart_machine, "[D", "getLikForTree", as.integer(t - 1))
#	}	
#	assign("all_tree_liks", all_tree_liks, .GlobalEnv)
#	
#	if (save_plot){
#		save_plot_function(bart_machine, "tree_liks_hist", data_title)
#	}	
#	else {
#		dev.new()
#	}	
#	
#	all_liks_after_burn_in = as.vector(all_tree_liks[, (num_burn_in + 1) : (num_gibbs + 1)])
#	min_lik = min(all_liks_after_burn_in)
#	max_lik = max(all_liks_after_burn_in)	
#	hist(all_liks_after_burn_in, 
#			col = "white", 
#			br = seq(from = min_lik - 0.01, to = max_lik + 0.01, by = (max_lik - min_lik + 0.01) / 100), 
#			border = NA, 
#			main = "Histogram of tree prop likelihoods after burn-in")
##	shapiro.test(all_liks_after_burn_in)
#	for (t in 1 : num_trees){
#		hist(all_tree_liks[t, (num_burn_in + 1) : (num_gibbs + 1)], 
#				col = rgb(runif(1, 0, 0.8), runif(1, 0, 0.8), runif(1, 0, 0.8), ALPHA), 
#				border = NA, 
#				br = seq(from = min_lik, to = max_lik, by = (max_lik - min_lik) / 100), 
#				add = TRUE)
#		
#	}
#	
#	if (save_plot){	
#		dev.off()
#	}	
#}

############# EXPERIMENTAL
#############


#get_variable_significance = function(bart_machine, var_num, data = NULL, num_iter = 100, print_histogram = TRUE, num_cores = 1){
#	if (bart_machine$run_in_sample){
#		real_sse = bart_machine$L2_err_train
#		n = bart_machine$n
#		data = bart_machine$training_data
#	} else {
#		real_sse = bart_predict_for_test_data(bart_machine, data, num_cores)$L2_err
#		n = nrow(data)
#	}
#	
#	
#	sse_vec = array(NA, num_iter)
#	
#	for (i in 1 : num_iter){
#		data_scrambled = data
#		data_scrambled[, var_num] = data[sample(1 : n, n, replace = FALSE), var_num] ##scrambled column of interest
#		sse_vec[i] = bart_predict_for_test_data(bart_machine, data_scrambled, num_cores)$L2_err
#		if (i %% 10 == 0) print(i)
#	}
#	p_value = (1 + sum(real_sse > sse_vec)) / (1 + num_iter) ##how many null values greater than obs. 
#	if (print_histogram){
#		windows()
#		hist(sse_vec, 
#				br = num_iter / 3,
#				col = "grey", 
#				main = paste("Null Distribution for Variable", var_num),
#				xlim = c(min(sse_vec,real_sse-1),max(sse_vec, real_sse+1)),
#				xlab = "SSE")
#		abline(v=real_sse, col="blue", lwd = 3)
#	}
#	list(p_value = p_value, sse_vec = sse_vec, real_sse = real_sse)
#}



#plot_sigsqs_convergence_diagnostics_hetero = function(bart_machine, records = c(1), extra_text = NULL, data_title = "data_model", save_plot = FALSE, moving_avgs = TRUE){
#	if (bart_machine$use_heteroskedasticity != TRUE){
#		stop("This BART machine was not created using the heteroskedasticity-robust feature, use \"plot_sigsqs_convergence_diagnostics\" instead")
#	}
#	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
#	
#	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
#	num_burn_in = bart_machine$num_burn_in
#	num_gibbs = bart_machine$num_gibbs
#	num_trees = bart_machine$num_trees
#	
#	sigsqs = matrix(NA, nrow = bart_machine$num_gibbs, ncol = bart_machine$n)
#	for (g in 1 : bart_machine$num_gibbs){
#		sigsqs[g, ] = .jcall(bart_machine$java_bart_machine, "[D", "getSigsqsByGibbsSample", as.integer(g - 1))
#	}
#	sigsqs_after_burnin = sigsqs[(length(sigsqs) - num_iterations_after_burn_in) : length(sigsqs), 1]
#	assign("sigsqs_after_burnin", sigsqs_after_burnin, .GlobalEnv)	
#	
#	if (save_plot){
#		save_plot_function(bart_machine, "sigsqs_by_gibbs", data_title)
#	}
#	else {
#		dev.new()
#	}	
#	
#	ymax = quantile(sigsqs, .95)
##	ymax = max(sigsqs[bart_machine$num_burn_in : bart_machine$num_gibbs, ])
#	
#	plot(NA, 
#			main = paste("Sigsqs throughout entire chain", ifelse(is.null(extra_text), "", 
#							paste("\n", extra_text)), ifelse(length(records) < 10, paste("record #", paste(records, collapse = ", ")), "")), 
#			type = "n", 
#			xlab = "Gibbs sample",
#			ylab = "Var[Noise] estimate",
#			xlim = c(1, bart_machine$num_gibbs), 		
#			ylim = c(0, ymax)
#	)
#	
#	#want to plot each sigsq as a function of gibbs
#	for (i in records){
#		sigsqis = sigsqs[, i]
#		points(sigsqis, pch = ".", col = COLORS[i %% 500])
#	}
#	abline(v = num_burn_in, col = "gray")
#	
#	#now maybe we want to see moving averages
#	if (moving_avgs){
#		for (i in records){
#			sigsqis = sigsqs[, i]
#			moving_avg = filter(sigsqis, rep(1/101, 101), sides = 2)
#			lines(moving_avg, col = COLORS[i %% 500])
#		}		
#	}
#	
#	if (save_plot){	
#		dev.off()
#	}
#	
#	sigsqs
#}


plot_y_vs_yhat_with_predictions = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	if (bart_machine$run_in_sample == FALSE){
		stop("you can only plot after running in-sample\n")
	}
	#TODO needs to be done
	y = bart_machine
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


plot_all_mu_values_for_tree = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE, tree_num){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}		
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
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}		
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
