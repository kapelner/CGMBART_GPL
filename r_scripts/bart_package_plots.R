
plot_tree_depths = function(bart_machine){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	
	tree_depths_after_burn_in = get_tree_depths(bart_machine)
	
	num_after_burn_in_per_core = ncol(tree_depths_after_burn_in)
	
	plot(1 : num_after_burn_in_per_core, rep(0, num_after_burn_in_per_core), type = "n", 
		main = "Tree Depth by Gibbs Sample After Burn-in", xlab = "Gibbs Sample", 
		ylab = paste("Tree Depth for all cores (", nrow(tree_depths_after_burn_in), " trees)", sep = ""), ylim = c(0, max(tree_depths_after_burn_in)))
	#plot burn in
	for (t in 1 : nrow(tree_depths_after_burn_in)){
		lines(1 : num_after_burn_in_per_core, tree_depths_after_burn_in[t, ], col = rgb(0.9,0.9,0.9))
	}
	lines(1 : num_after_burn_in_per_core, apply(tree_depths_after_burn_in, 2, mean), col = "blue", lwd = 4)
	lines(1 : num_after_burn_in_per_core, apply(tree_depths_after_burn_in, 2, min), col = "black")
	lines(1 : num_after_burn_in_per_core, apply(tree_depths_after_burn_in, 2, max), col = "black")
}

get_tree_depths = function(bart_machine){
	tree_depths_after_burn_in = NULL
	for (c in 1 : bart_machine$num_cores){
		tree_depths_after_burn_in_core = sapply(.jcall(bart_machine$java_bart_machine, "[[I", "getDepthsForTreesInGibbsSampAfterBurnIn", as.integer(c)), .jevalArray)
		tree_depths_after_burn_in = rbind(tree_depths_after_burn_in, tree_depths_after_burn_in_core)
	}
	tree_depths_after_burn_in
}

plot_tree_num_nodes = function(bart_machine){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	
	tree_num_nodes_and_leaves_after_burn_in = get_tree_num_nodes_and_leaves(bart_machine)
	
	num_after_burn_in_per_core = ncol(tree_num_nodes_and_leaves_after_burn_in)
	
	plot(1 : num_after_burn_in_per_core, rep(0, num_after_burn_in_per_core), type = "n", 
		main = "Tree Num Nodes And Leaves by Gibbs Sample After Burn-in", xlab = "Gibbs Sample", 
		ylab = paste("Tree Num Nodes and Leaves for all cores (", nrow(tree_num_nodes_and_leaves_after_burn_in), " trees)", sep = ""), 
		ylim = c(0, max(tree_num_nodes_and_leaves_after_burn_in)))
	#plot burn in
	for (t in 1 : nrow(tree_num_nodes_and_leaves_after_burn_in)){
		lines(1 : num_after_burn_in_per_core, tree_num_nodes_and_leaves_after_burn_in[t, ], col = rgb(0.9,0.9,0.9))
	}
	lines(1 : num_after_burn_in_per_core, apply(tree_num_nodes_and_leaves_after_burn_in, 2, mean), col = "blue", lwd = 4)
	lines(1 : num_after_burn_in_per_core, apply(tree_num_nodes_and_leaves_after_burn_in, 2, min), col = "black")
	lines(1 : num_after_burn_in_per_core, apply(tree_num_nodes_and_leaves_after_burn_in, 2, max), col = "black")
}

get_tree_num_nodes_and_leaves = function(bart_machine){
	tree_num_nodes_and_leaves_after_burn_in = NULL
	for (c in 1 : bart_machine$num_cores){
		tree_num_nodes_and_leaves_after_burn_in_core = sapply(.jcall(bart_machine$java_bart_machine, "[[I", "getNumNodesAndLeavesForTreesInGibbsSampAfterBurnIn", as.integer(c)), .jevalArray)
		tree_num_nodes_and_leaves_after_burn_in = rbind(tree_num_nodes_and_leaves_after_burn_in, tree_num_nodes_and_leaves_after_burn_in_core)
	}
	tree_num_nodes_and_leaves_after_burn_in
}

plot_mh_acceptance_reject = function(bart_machine){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	
	mh_acceptance_reject = get_mh_acceptance_reject(bart_machine)
	a_r_before_burn_in = mh_acceptance_reject[["a_r_before_burn_in"]]
	a_r_before_burn_in_avg_over_trees = rowSums(a_r_before_burn_in) / bart_machine$num_trees
	
	a_r_after_burn_in_avgs_over_trees = list()
	for (c in 1 : bart_machine$num_cores){
		a_r_after_burn_in = mh_acceptance_reject[["a_r_after_burn_in"]][[c]]
		a_r_after_burn_in_avgs_over_trees[[c]] = rowSums(a_r_after_burn_in) / bart_machine$num_trees		
	}	
	
	num_after_burn_in_per_core = length(a_r_after_burn_in_avgs_over_trees[[1]])
	num_gibbs_per_core = bart_machine$num_burn_in + num_after_burn_in_per_core
	
	
	plot(1 : num_gibbs_per_core, rep(0, num_gibbs_per_core), ylim = c(0, 1), type = "n", 
			main = "Percent Acceptance by Gibbs Sample", xlab = "Gibbs Sample", ylab = "% of Trees Accepting")
	abline(v = bart_machine$num_burn_in, col = "grey")
	#plot burn in
	points(1 : bart_machine$num_burn_in, a_r_before_burn_in_avg_over_trees, col = "grey")
	lines(loess.smooth(1 : bart_machine$num_burn_in, a_r_before_burn_in_avg_over_trees), col = "black", lwd = 4)
	
	for (c in 1 : bart_machine$num_cores){
		points((bart_machine$num_burn_in + 1) : num_gibbs_per_core, a_r_after_burn_in_avgs_over_trees[[c]], col = COLORS[c])
		lines(loess.smooth((bart_machine$num_burn_in + 1) : num_gibbs_per_core, a_r_after_burn_in_avgs_over_trees[[c]]), col = COLORS[c], lwd = 4)
	}	
	
}


get_mh_acceptance_reject = function(bart_machine){
	a_r_before_burn_in = t(sapply(.jcall(bart_machine$java_bart_machine, "[[Z", "getAcceptRejectMHsBurnin"), .jevalArray)) * 1
	
	a_r_after_burn_in = list()
	for (c in 1 : bart_machine$num_cores){
		a_r_after_burn_in[[c]] = t(sapply(.jcall(bart_machine$java_bart_machine, "[[Z", "getAcceptRejectMHsAfterBurnIn", as.integer(c)), .jevalArray)) * 1
	}
	
	list(
			a_r_before_burn_in = a_r_before_burn_in,
			a_r_after_burn_in = a_r_after_burn_in	
	)
}

plot_y_vs_yhat = function(bart_machine, ppis = FALSE, ppi_conf = 0.95, num_cores = 1){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	y_hat = bart_machine$y_hat_train
	y = bart_machine$y_train

	
	if (ppis){
		ppis = calc_ppis_from_prediction(bart_machine, bart_machine$training_data, ppi_conf, num_cores)
		ppi_a = ppis[, 1]
		ppi_b = ppis[, 2]
		y_in_ppi = y >= ppi_a & y <= ppi_b
		prop_ys_in_ppi = sum(y_in_ppi) / length(y_in_ppi)
		
		plot(y, y_hat, 
			main = paste("Fitted vs. Actual Values with ", round(ppi_conf * 100), "% PPIs (", round(prop_ys_in_ppi * 100, 2), "% coverage)", sep = ""), 
			xlab = paste("Actual Values", sep = ""), 
			ylab = "Fitted Values", 
			xlim = c(min(min(y), min(y_hat)), max(max(y), max(y_hat))),
			ylim = c(min(min(y), min(y_hat)), max(max(y), max(y_hat))),
			cex = 0)
		#draw PPI's
		for (i in 1 : bart_machine$n){
			segments(y[i], ppi_a[i], y[i], ppi_b[i], col = "grey", lwd = 0.1)	
		}
		#draw green dots or red dots depending upon inclusion in the PPI
		for (i in 1 : bart_machine$n){
			points(y[i], y_hat[i], col = ifelse(y_in_ppi[i], "darkgreen", "red"), cex = 0.6, pch = 16)	
		}		
	} else {
		plot(y, y_hat, main = "Fitted vs. Actual Values", xlab = "Actual Values", ylab = "Fitted Values", col = "blue")
	}
	abline(a = 0, b = 1, lty = 2)	
}

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


hist_sigsqs = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
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


plot_sigsqs_convergence_diagnostics = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
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

investigate_var_importance = function(bart_machine, plot = TRUE, use_bottleneck = TRUE, num_replicates_for_avg = 10, num_trees_bottleneck = 20){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	
	avg_var_props = get_averaged_true_var_props(bart_machine, use_bottleneck, num_replicates_for_avg, num_trees_bottleneck)
	
	if (plot){
		barplot(avg_var_props, names = names(avg_var_props), las = 2, main = paste("Important Variables"), xlab = "Variable", ylab = "Inclusion Proportion")	
	}
	
	avg_var_props
}

get_averaged_true_var_props = function(bart_machine, use_bottleneck, num_replicates_for_avg, num_trees_bottleneck){
	if (!use_bottleneck){
		get_var_props_over_chain(bart_machine)
	}
	else {
		var_props = rep(0, ncol(bart_machine$training_data) - 1)
		for (i in 1 : num_replicates_for_avg){
			bart_machine_dup = build_bart_machine(bart_machine$training_data, 
				num_trees = num_trees_bottleneck, 
				num_burn_in = bart_machine$num_burn_in, 
				num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in, 
				cov_prior_vec = bart_machine$cov_prior_vec,
				run_in_sample = FALSE,
				verbose = FALSE)
			var_props = var_props + get_var_props_over_chain(bart_machine_dup)
			destroy_bart_machine(bart_machine_dup)
			cat(".")
		}
		cat("\n")
		#average over many runs
		var_props / num_replicates_for_avg		
	}
}