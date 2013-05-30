

check_bart_error_assumptions = function(bart_machine, alpha_normal_test = 0.05, alpha_hetero_test = 0.05, hetero_plot = "yhats"){
	#load the library
	tryCatch(library(car), error = function(e){install.packages("car")}, finally = library(car))
	
	if (bart_machine$pred_type == "classification"){
		stop("There are no convergence diagnostics for classification.")
	}	
	graphics.off()
	par(mfrow = c(1, 2))
	es = bart_machine$residuals
	y_hat = bart_machine$y_hat
	
	#test for normality
	normal_p_val = shapiro.test(es)$p.value
	qqp(es, col = "blue",
			main = paste("Assessment of Normality\n", "p-val for shapiro-wilk test of normality of residuals:", round(normal_p_val, 3)),
			xlab = "Normal Q-Q plot for in-sample residuals\n(Theoretical Quantiles)")	
	
	#test for heteroskedasticity
	if (hetero_plot == "yhats"){
		plot(y_hat, es, main = paste("Assessment of Heteroskedasticity\nFitted vs residuals"), xlab = "Fitted Values", ylab = "Residuals", col = "blue")
	} else {
		plot(bart_machine$y, es, main = paste("Assessment of Heteroskedasticity\nFitted vs residuals"), xlab = "Actual Values", ylab = "Residuals", col = "blue")
	}
	
	abline(h = 0, col = "black")
#	cat("p-val for shapiro-wilk test of normality of residuals:", normal_p_val, ifelse(normal_p_val > alpha_normal_test, "(ppis believable)", "(exercise caution when using ppis!)"), "\n")
	par(mfrow = c(1, 1))
	#TODO --- iterate over all x's and sort them
	#see p225 in purple book for Szroeter's test
#	n = length(es)
#	h = sum(seq(1 : n) * es^2) / sum(es^2)
#	Q = sqrt(6 * n / (n^2 - 1)) * (h - (n + 1) / 2)
#	hetero_pval = 1 - pnorm(Q, 0, 1)
#	cat("p-val for szroeter's test of homoskedasticity of residuals (assuming inputted observation order):", hetero_pval, ifelse(hetero_pval > alpha_hetero_test, "(ppis believable)", "(exercise caution when using ppis!)"), "\n")		
	
}

plot_tree_depths = function(bart_machine){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	
	tree_depths_after_burn_in = get_tree_depths(bart_machine)
	
	num_after_burn_in_per_core = nrow(tree_depths_after_burn_in)
	
	plot(1 : num_after_burn_in_per_core, rep(0, num_after_burn_in_per_core), type = "n", 
		main = "Tree Depth by Gibbs Sample After Burn-in", xlab = "Gibbs Sample", 
		ylab = paste("Tree Depth for all cores"), ylim = c(0, max(tree_depths_after_burn_in)))
	#plot burn in
	for (t in 1 : ncol(tree_depths_after_burn_in)){
		lines(1 : num_after_burn_in_per_core, tree_depths_after_burn_in[, t], col = rgb(0.9,0.9,0.9))
	}
	lines(1 : num_after_burn_in_per_core, apply(tree_depths_after_burn_in, 1, mean), col = "blue", lwd = 4)
	lines(1 : num_after_burn_in_per_core, apply(tree_depths_after_burn_in, 1, min), col = "black")
	lines(1 : num_after_burn_in_per_core, apply(tree_depths_after_burn_in, 1, max), col = "black")
	
	
	if (bart_machine$num_cores > 1){
		for (c in 2 : bart_machine$num_cores){
			abline(v = (c - 1) * bart_machine$num_iterations_after_burn_in / bart_machine$num_cores, col = "gray")
		}		
	}		
}

get_tree_depths = function(bart_machine){
	tree_depths_after_burn_in = NULL
	for (c in 1 : bart_machine$num_cores){
		tree_depths_after_burn_in_core = t(sapply(.jcall(bart_machine$java_bart_machine, "[[I", "getDepthsForTreesInGibbsSampAfterBurnIn", as.integer(c)), .jevalArray))
		tree_depths_after_burn_in = rbind(tree_depths_after_burn_in, tree_depths_after_burn_in_core)
	}
	tree_depths_after_burn_in
}

plot_tree_num_nodes = function(bart_machine){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	
	tree_num_nodes_and_leaves_after_burn_in = get_tree_num_nodes_and_leaves(bart_machine)
	
	num_after_burn_in_per_core = nrow(tree_num_nodes_and_leaves_after_burn_in)
	
	plot(1 : num_after_burn_in_per_core, rep(0, num_after_burn_in_per_core), type = "n", 
		main = "Tree Num Nodes And Leaves by Gibbs Sample After Burn-in", xlab = "Gibbs Sample", 
		ylab = paste("Tree Num Nodes and Leaves for all cores"), 
		ylim = c(0, max(tree_num_nodes_and_leaves_after_burn_in)))
	#plot burn in
	for (t in 1 : ncol(tree_num_nodes_and_leaves_after_burn_in)){
		lines(1 : num_after_burn_in_per_core, tree_num_nodes_and_leaves_after_burn_in[, t], col = rgb(0.9, 0.9, 0.9))
	}
	lines(1 : num_after_burn_in_per_core, apply(tree_num_nodes_and_leaves_after_burn_in, 1, mean), col = "blue", lwd = 4)
	lines(1 : num_after_burn_in_per_core, apply(tree_num_nodes_and_leaves_after_burn_in, 1, min), col = "black")
	lines(1 : num_after_burn_in_per_core, apply(tree_num_nodes_and_leaves_after_burn_in, 1, max), col = "black")
	
	if (bart_machine$num_cores > 1){
		for (c in 2 : bart_machine$num_cores){
			abline(v = (c - 1) * bart_machine$num_iterations_after_burn_in / bart_machine$num_cores, col = "gray")
		}		
	}	
}

get_tree_num_nodes_and_leaves = function(bart_machine){
	tree_num_nodes_and_leaves_after_burn_in = NULL
	for (c in 1 : bart_machine$num_cores){
		tree_num_nodes_and_leaves_after_burn_in_core = t(sapply(.jcall(bart_machine$java_bart_machine, "[[I", "getNumNodesAndLeavesForTreesInGibbsSampAfterBurnIn", as.integer(c)), .jevalArray))
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
	tryCatch(lines(loess.smooth(1 : bart_machine$num_burn_in, a_r_before_burn_in_avg_over_trees), col = "black", lwd = 4), error = function(e){e})
	
	for (c in 1 : bart_machine$num_cores){
		points((bart_machine$num_burn_in + 1) : num_gibbs_per_core, a_r_after_burn_in_avgs_over_trees[[c]], col = COLORS[c])
		tryCatch(lines(loess.smooth((bart_machine$num_burn_in + 1) : num_gibbs_per_core, a_r_after_burn_in_avgs_over_trees[[c]]), col = COLORS[c], lwd = 4), error = function(e){e})
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

plot_y_vs_yhat = function(bart_machine, X = NULL, y = NULL, ppis = FALSE, ppi_conf = 0.95){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	
	if (is.null(X) & is.null(y)){
		X = bart_machine$X
		y = bart_machine$y
		y_hat = bart_machine$y_hat_train
		in_sample = TRUE
	} else {
		predict_obj = bart_predict_for_test_data(bart_machine, X, y)
		y_hat = predict_obj$y_hat
		in_sample = FALSE
	}
	
	if (ppis){
		ppis = calc_ppis_from_prediction(bart_machine, X, ppi_conf)
		ppi_a = ppis[, 1]
		ppi_b = ppis[, 2]
		y_in_ppi = y >= ppi_a & y <= ppi_b
		prop_ys_in_ppi = sum(y_in_ppi) / length(y_in_ppi)
		
		plot(y, y_hat, 
			main = paste(ifelse(in_sample, "In-Sample", "Out-of-Sample"), " Fitted vs. Actual Values with ", round(ppi_conf * 100), "% PPIs (", round(prop_ys_in_ppi * 100, 2), "% coverage)", sep = ""), 
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
		plot(y, y_hat, 
			main = "Fitted vs. Actual Values", 
			xlab = "Actual Values", 
			ylab = "Fitted Values", 
			col = "blue", 
			xlim = c(min(min(y), min(y_hat)), max(max(y), max(y_hat))),
			ylim = c(min(min(y), min(y_hat)), max(max(y), max(y_hat))),)
	}
	abline(a = 0, b = 1, lty = 2)	
}



hist_sigsqs = function(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	if (bart_machine$pred_type == "classification"){
		stop("There are no convergence diagnostics for classification.")
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
	
	invisible(sigsqs_after_burnin)
}

get_sigsqs = function(bart_machine, after_burn_in = TRUE){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
	
	if (after_burn_in){
		num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
		num_burn_in = bart_machine$num_burn_in
		num_gibbs = bart_machine$num_gibbs
		num_trees = bart_machine$num_trees
		
		sigsqs[(length(sigsqs) - num_iterations_after_burn_in) : length(sigsqs)]
	} else {
		sigsqs
	}
	
}

plot_sigsqs_convergence_diagnostics = function(bart_machine){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	if (bart_machine$pred_type == "classification"){
		stop("There are no convergence diagnostics for classification.")
	}	
	
	sigsqs = get_sigsqs(bart_machine, after_burn_in = FALSE)
	
	num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
	num_burn_in = bart_machine$num_burn_in
	num_gibbs = bart_machine$num_gibbs
	num_trees = bart_machine$num_trees
	
	#first look at sigsqs
	sigsqs_after_burnin = sigsqs[(length(sigsqs) - num_iterations_after_burn_in) : length(sigsqs)]
	avg_sigsqs_after_burn_in = mean(sigsqs_after_burnin, na.rm = TRUE)
	
	plot(sigsqs, 
		main = paste("Sigsq Estimates over Gibbs Samples"), 
		xlab = "Gibbs sample (yellow lines: after burn-in 95% PPI)", 
		ylab = paste("Sigsq by iteration, avg after burn-in =", round(avg_sigsqs_after_burn_in, 3)),
		ylim = c(quantile(sigsqs, 0.01), quantile(sigsqs, 0.99)),
		pch = ".", 
		cex = 3,
		col = "gray")
	points(sigsqs, pch = ".", col = "red")
	ppi_sigsqs = quantile(sigsqs[num_burn_in : length(sigsqs)], c(.025, .975))
	abline(a = ppi_sigsqs[1], b = 0, col = "yellow")
	abline(a = ppi_sigsqs[2], b = 0, col = "yellow")
	abline(a = avg_sigsqs_after_burn_in, b = 0, col = "blue")
	abline(v = num_burn_in, col = "gray")
	if (bart_machine$num_cores > 1){
		for (c in 2 : bart_machine$num_cores){
			abline(v = num_burn_in + (c - 1) * bart_machine$num_iterations_after_burn_in / bart_machine$num_cores, col = "gray")
		}		
	}

}

investigate_var_importance = function(bart_machine, type = "splits", plot = TRUE, num_replicates_for_avg = 5, num_trees_bottleneck = 20, num_var_plot = Inf){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}	

	var_props = array(0, c(num_replicates_for_avg, bart_machine$p))
	for (i in 1 : num_replicates_for_avg){
		if (i == 1 & num_trees_bottleneck == bart_machine$num_trees){
			var_props[i, ] = get_var_props_over_chain(bart_machine, type)
		} else {
			bart_machine_dup = build_bart_machine(bart_machine$X, bart_machine$y, 
				num_trees = num_trees_bottleneck, 
				num_burn_in = bart_machine$num_burn_in, 
				num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in, 
				cov_prior_vec = bart_machine$cov_prior_vec,
				run_in_sample = FALSE,
				verbose = FALSE)			
			var_props[i, ] = get_var_props_over_chain(bart_machine_dup, type)
			destroy_bart_machine(bart_machine_dup)						
		}
		cat(".")
	}
	cat("\n")
	
	avg_var_props = colMeans(var_props)
	names(avg_var_props) = bart_machine$training_data_features_with_missing_features
	sd_var_props = apply(var_props, 2, sd)
	names(sd_var_props) = bart_machine$training_data_features_with_missing_features
	
	if (num_var_plot == Inf){
		num_var_plot = bart_machine$p
	}
	
	avg_var_props_sorted_indices = sort(avg_var_props, decreasing = TRUE, index.return = TRUE)$ix
	avg_var_props = avg_var_props[avg_var_props_sorted_indices][1 : num_var_plot]
	sd_var_props = sd_var_props[avg_var_props_sorted_indices][1 : num_var_plot]		
	
	if (plot){
		par(mar = c(5, 6, 3, 0))
		if (is.na(sd_var_props[1])){
			moe = 0
		} else {
			moe = 1.96 * sd_var_props / sqrt(num_replicates_for_avg)
		}
		bars = barplot(avg_var_props, 
				names.arg = names(avg_var_props), 
				las = 2, 
				ylab = "Inclusion Proportion", 
				col = "gray",#rgb(0.39, 0.39, 0.59),
				ylim = c(0, max(avg_var_props + moe)),
				main = paste("Important Variables Averaged over", num_replicates_for_avg, "Replicates by", ifelse(type == "splits", "Number of Variable Splits", "Number of Trees")))
		conf_upper = avg_var_props + 1.96 * sd_var_props / sqrt(num_replicates_for_avg)
		conf_lower = avg_var_props - 1.96 * sd_var_props / sqrt(num_replicates_for_avg)
		segments(bars, avg_var_props, bars, conf_upper, col = rgb(0.59, 0.39, 0.39), lwd = 3) # Draw error bars
		segments(bars, avg_var_props, bars, conf_lower, col = rgb(0.59, 0.39, 0.39), lwd = 3)		
	}
	
	invisible(list(avg_var_props = avg_var_props, sd_var_props = sd_var_props))	
}

plot_convergence_diagnostics = function(bart_machine){
	par(mfrow = c(2, 2))
	if (bart_machine$pred_type == "regression"){
		plot_sigsqs_convergence_diagnostics(bart_machine)
	}	
	plot_mh_acceptance_reject(bart_machine)
	plot_tree_num_nodes(bart_machine)
	plot_tree_depths(bart_machine)	
	par(mfrow = c(1, 1))
}

shapiro_wilk_p_val = function(vec){
	tryCatch(shapiro.test(vec)$p.value, error = function(e){})
}

interaction_investigator = function(bart_machine, plot = TRUE, num_replicates_for_avg = 5, num_trees_bottleneck = 20, num_var_plot = Inf, cut_bottom = NULL){
	
	interaction_counts = array(NA, c(bart_machine$p, bart_machine$p, num_replicates_for_avg))
	
	for (r in 1 : num_replicates_for_avg){
		if (r == 1 & num_trees_bottleneck == bart_machine$num_trees){
			interaction_counts[, , r] = sapply(.jcall(bart_machine$java_bart_machine, "[[I", "getInteractionCounts", as.integer(BART_NUM_CORES)), .jevalArray)
		} else {
			bart_machine_dup = bart_machine_duplicate(bart_machine)			
			interaction_counts[, , r] = sapply(.jcall(bart_machine_dup$java_bart_machine, "[[I", "getInteractionCounts", as.integer(BART_NUM_CORES)), .jevalArray)
			destroy_bart_machine(bart_machine_dup)
			cat(".")
			if (r %% 40 == 0){
				cat("\n")
			}					
		}
	}
	cat("\n")
	
	interaction_counts_avg = apply(interaction_counts, 1 : 2, mean)
  if(bart_machine$use_missing_data == T){
    rownames(interaction_counts_avg) = bart_machine$training_data_features_with_missing_features
    colnames(interaction_counts_avg) = bart_machine$training_data_features_with_missing_features
  }
  else{
	  rownames(interaction_counts_avg) = bart_machine$training_data_features
	  colnames(interaction_counts_avg) = bart_machine$training_data_features	
  }
  interaction_counts_sd = apply(interaction_counts, 1 : 2, sd)
	interaction_counts_s_w_test = apply(interaction_counts, 1 : 2, shapiro_wilk_p_val)
	
	#now vectorize the interaction counts
	avg_counts = array(NA, bart_machine$p * (bart_machine$p - 1) / 2)
	sd_counts = array(NA, bart_machine$p * (bart_machine$p - 1) / 2)
	iter = 1
	for (i in 1 : bart_machine$p){
		for (j in 1 : bart_machine$p){
			if (j <= i){
				avg_counts[iter] = interaction_counts_avg[i, j]
				sd_counts[iter] = interaction_counts_sd[i, j]
				names(avg_counts)[iter] = paste(rownames(interaction_counts_avg)[i], "x", rownames(interaction_counts_avg)[j])
				iter = iter + 1
			}
		}
	}
	num_total_interactions = bart_machine$p * (bart_machine$p + 1) / 2
	if (num_var_plot == Inf || num_var_plot > num_total_interactions){
		num_var_plot = num_total_interactions
	}
	
	avg_counts_sorted_indices = sort(avg_counts, decreasing = TRUE, index.return = TRUE)$ix
	avg_counts = avg_counts[avg_counts_sorted_indices][1 : num_var_plot]
	sd_counts = sd_counts[avg_counts_sorted_indices][1 : num_var_plot]
	
	if (is.null(cut_bottom)){
		ylim_bottom = 0
	} else {
		ylim_bottom = cut_bottom * min(avg_counts)
	}
	
	##TO-DO: kill zeroes from the plots
	
	if (plot){
		#now create the bar plot
		par(mar = c(10, 6, 3, 0))
		if (is.na(sd_counts[1])){
			moe = 0
		} else {
			moe = 1.96 * sd_counts / sqrt(num_replicates_for_avg)
		}
		bars = barplot(avg_counts, 
			names.arg = names(avg_counts), 
			las = 2, 
			ylab = "Relative Importance", 
			col = "gray",#rgb(0.39, 0.39, 0.59),
			ylim = c(ylim_bottom, max(avg_counts + moe)),
			xpd = FALSE, #clips the bars outside of the display region (why is this not a default setting?)
			main = paste("Interactions in BART Model Averaged over", num_replicates_for_avg, "Replicates"))
		if (!is.na(sd_counts[1])){
			conf_upper = avg_counts + 1.96 * sd_counts / sqrt(num_replicates_for_avg)
			conf_lower = avg_counts - 1.96 * sd_counts / sqrt(num_replicates_for_avg)
			segments(bars, avg_counts, bars, conf_upper, col = rgb(0.59, 0.39, 0.39), lwd = 3) # Draw error bars
			segments(bars, avg_counts, bars, conf_lower, col = rgb(0.59, 0.39, 0.39), lwd = 3)			
		}		
	}
	
	invisible(list(
		interaction_counts_avg = interaction_counts_avg, 
		interaction_counts_sd = interaction_counts_sd, 
		interaction_counts_s_w_test = interaction_counts_s_w_test
	))
}

pd_plot = function(bart_machine, j, levs = c(0.05, seq(from = 0.10, to = 0.90, by = 0.10), 0.95), lower_ci = 0.05, upper_ci = 0.95){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	if (j < 1 || j > bart_machine$p){
		stop(paste("You must set j to a number between 1 and p =", bart_machine$p))
	}
	
	x_j = bart_machine$X[, j]
	x_j_quants = quantile(x_j, levs)
	bart_predictions_by_quantile = array(NA, c(length(levs), bart_machine$n, bart_machine$num_iterations_after_burn_in))
	
	for (q in 1 : length(levs)){
		x_j_quant = x_j_quants[q]
		
		#now create test data matrix
		test_data = bart_machine$X
		test_data[, j] = rep(x_j_quant, bart_machine$n)
		
		bart_predictions_by_quantile[q, , ] = bart_machine_predict(bart_machine, test_data)$y_hat_posterior_samples
		cat(".")
	}
	cat("\n")
	
	bart_avg_predictions_by_quantile_by_gibbs = array(NA, c(length(levs), bart_machine$num_iterations_after_burn_in))
	for (q in 1 : length(levs)){
		for (g in 1 : bart_machine$num_iterations_after_burn_in){
			bart_avg_predictions_by_quantile_by_gibbs[q, g] = mean(bart_predictions_by_quantile[q, , g])
		}
		
	}
	
	bart_avg_predictions_by_quantile = apply(bart_avg_predictions_by_quantile_by_gibbs, 1, mean)
	bart_avg_predictions_lower = apply(bart_avg_predictions_by_quantile_by_gibbs, 1, quantile, probs = lower_ci)
	bart_avg_predictions_upper = apply(bart_avg_predictions_by_quantile_by_gibbs, 1, quantile, probs = upper_ci)
	
	plot(x_j_quants, bart_avg_predictions_by_quantile, 
			type = "o", 
			main = "Partial Dependence Plot",
			ylim = c(min(bart_avg_predictions_lower, bart_avg_predictions_upper), max(bart_avg_predictions_lower, bart_avg_predictions_upper)),
			ylab = "Partial Effect",
			xlab = paste("Quantiles for Variable", bart_machine$training_data_features[j]))
	lines(x_j_quants, bart_avg_predictions_lower, type = "o", col = "blue")
	lines(x_j_quants, bart_avg_predictions_upper, type = "o", col = "blue")
	
	invisible(list(x_j_quants = x_j_quants, bart_avg_predictions_by_quantile = bart_avg_predictions_by_quantile))
#	rob = bart(x.train = bart_machine$X, y.train = bart_machine$y, ndpost = 100, nskip = 500, keepevery = 10)
#	pdbart(x.train = bart_machine$X, y.train = bart_machine$y, xind = 6, ndpost = 200, nskip = 500)
}

rmse_by_num_trees = function(bart_machine, tree_list = c(1, seq(5, 50, 5), 100, 150, 200, 300), in_sample = FALSE, plot = TRUE, holdout_pctg = 0.3, num_replicates = 4){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	if (bart_machine$pred_type == "classification"){
		stop("This function does not work for classification. Please use ")
	}		
	X = bart_machine$X
	y = bart_machine$y
	n = bart_machine$n
	
	rmses = array(NA, c(num_replicates, length(tree_list)))
	cat("num_trees = ")
	for (t in 1 : length(tree_list)){
		for (r in 1 : num_replicates){
			if (in_sample){
				bart_machine_dup = bart_machine_duplicate(bart_machine, num_trees = tree_list[t], run_in_sample = TRUE)
				rmses[r, t] = bart_machine_dup$rmse_train				
			} else {
				holdout_indicies = sample(1 : n, holdout_pctg * n)
				Xtrain = X[setdiff(1 : n, holdout_indicies), ]
				ytrain = y[setdiff(1 : n, holdout_indicies)]
				Xtest = X[holdout_indicies, ]
				ytest = y[holdout_indicies]
				
				bart_machine_dup = bart_machine_duplicate(bart_machine, Xtrain, ytrain, num_trees = tree_list[t])
				predict_obj = bart_predict_for_test_data(bart_machine_dup, Xtest, ytest)
				rmses[r, t] = predict_obj$rmse				
			}
			destroy_bart_machine(bart_machine_dup)
			cat("..")
			cat(tree_list[t])			
		}
	}
	cat("\n")
	
	rmse_means = colMeans(rmses)

	if (plot){
		rmse_sds = apply(rmses, 2, sd)
		y_mins = rmse_means - 2 * rmse_sds
		y_maxs = rmse_means + 2 * rmse_sds
		plot(tree_list, rmse_means, 
			type = "o", 
			xlab = "Number of Trees", 
			ylab = paste(ifelse(in_sample, "In-Sample", "Out-Of-Sample"), "RMSE"), 
			main = paste("Fit by Number of Trees", ifelse(in_sample, "In-Sample", "Out-Of-Sample")), 
			ylim = c(min(y_mins), max(y_maxs)))
		if (num_replicates > 1){
			for (t in 1 : length(tree_list)){
				segments(tree_list[t], rmse_means[t] - 1.96 * rmse_sds[t], tree_list[t], rmse_means[t] + 1.96 * rmse_sds[t], col = "grey", lwd = 0.1)
			}
		}
	}
	invisible(rmse_means)
}

