

directory_where_code_is = getwd() #usually we're on a linux box and we'll just navigate manually to the directory
#if we're on windows, then we're on the dev box, so use a prespecified directory
if (.Platform$OS.type == "windows"){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
}
setwd(directory_where_code_is)

source("r_scripts/bart_package.R")
source("r_scripts/create_simulated_models.R")
graphics.off()

simulation_results = matrix(NA, nrow = 0, ncol = 18)
colnames(simulation_results) = c(
		"data_model", 
		"m",
		"N_B", 
		"N_G", 
		"alpha", 
		"beta", 		
		"A_BART_L1", 
		"A_BART_L2", 
		"A_BART_rmse", 
		"R_BART_L1", 
		"R_BART_L2", 
		"R_BART_rmse",
		"RF_L1",
		"RF_L2",
		"RF_rmse",
		"CART_L1",
		"CART_L2",
		"CART_rmse"
)

avg_simulation_results = matrix(NA, nrow = 0, ncol = 13)
colnames(avg_simulation_results) = c(
		"data_model", 
		"m",
		"N_B", 
		"N_G",
		"alpha", 
		"beta", 		
		"A_BART_rmse_avg", 
		"A_BART_rmse_se", 
		"R_BART_rmse_avg",
		"R_BART_rmse_se",
		"pval",		
		"RF_rmse_avg",
		"RF_rmse_se"
)



#avg_simulation_results_pretty

run_bakeoff = function(){
	current_run = 0
	for (alpha in alphas_of_interest){
		for (beta in betas_of_interest){
			for (num_burn_in in num_burn_ins_of_interest){
				for (num_iterations_after_burn_in in num_iterations_after_burn_ins_of_interest){
					for (num_trees in num_trees_of_interest){
						graphics.off() #make sure to shut graphics off otherwise it eventually overloads the console
						
						for (real_regression_data_set in real_regression_data_sets){
							raw_data = read.csv(paste("datasets//", real_regression_data_set, ".csv", sep = ""))				
							#now pull out half training and half test *randomly*			
							training_indices = sort(sample(1 : nrow(raw_data), nrow(raw_data) / 2))
							test_indices = setdiff(1 : nrow(raw_data), training_indices)
							training_data = raw_data[training_indices, ]
							test_data = raw_data[test_indices, ]							
							for (mod in 1 : run_model_N_times){
								current_run = current_run + 1
								append_to_log(paste("starting model ", current_run, "\\", total_num_runs, "  \"", real_regression_data_set, "\", m = ", num_trees, ", n_B = ", num_burn_in, ", n_G_a = ", num_iterations_after_burn_in, " alpha = ", alpha, " beta = ", beta, sep = ""))
								run_bart_model_and_save_diags_and_results(training_data, test_data, real_regression_data_set, num_trees, num_burn_in, num_iterations_after_burn_in, alpha, beta)
							}
						}
		
						for (simulated_data_set in simulated_data_sets){
							training_data = simulate_data_from_simulation_name(simulated_data_set)
							test_data = simulate_data_from_simulation_name(simulated_data_set)							
							for (mod in 1 : run_model_N_times){
								current_run = current_run + 1
								append_to_log(paste("starting model ", current_run, "\\", total_num_runs, "  \"", simulated_data_set, "\", m = ", num_trees, ", n_B = ", num_burn_in, ", n_G_a = ", num_iterations_after_burn_in, " alpha = ", alpha, " beta = ", beta, sep = ""))
								run_bart_model_and_save_diags_and_results(training_data, test_data, simulated_data_set, num_trees, num_burn_in, num_iterations_after_burn_in, alpha, beta)
							}
						}
					}
				}
			}
		}
	}
	prettify_simulation_results_and_save_as_csv()
	create_avg_sim_results_and_save_as_csv()
	draw_boxplots_of_sim_results()
	calculate_cochran_global_pval()
}

calculate_cochran_global_pval = function(){
	n = nrow(avg_simulation_results)
	chi_sq = sum(-2 * log(avg_simulation_results_pretty$pval))
	1 - pchisq(chi_sq, 2 * n)
}

prettify_simulation_results_and_save_as_csv = function(){
	#now update simulation results object
	rownames(simulation_results) = NULL
	simulation_results = as.data.frame(simulation_results)
	for (j in 2 : 16){
		simulation_results[, j] = as.numeric(as.character(simulation_results[, j]))
	}
	#assign it to the object
	assign("simulation_results_pretty", simulation_results, .GlobalEnv)
	#write it to file
	write.csv(simulation_results, paste(PLOTS_DIR, "/", "simulation_results.csv", sep = ""), row.names = FALSE)
}

draw_boxplots_of_sim_results = function(){
	graphics.off() #just clear it out first
	for (alpha in alphas_of_interest){
		for (beta in betas_of_interest){	
			for (num_burn_in in num_burn_ins_of_interest){
				for (num_iterations_after_burn_in in num_iterations_after_burn_ins_of_interest){
					for (num_trees in num_trees_of_interest){				
						for (data_set in c(real_regression_data_sets, simulated_data_sets)){
							draw_one_boxplot_and_save(data_set, num_trees, num_iterations_after_burn_in, num_burn_in, alpha, beta)
						}
					}
				}
			}
		}
	}
}

draw_one_boxplot_and_save = function(data_set, num_trees, num_iterations_after_burn_in, num_burn_in, alpha, beta){
	all_results = simulation_results_pretty[simulation_results_pretty$data_model == data_set & simulation_results_pretty$m == num_trees & simulation_results_pretty$N_B == num_burn_in & simulation_results_pretty$N_G == num_iterations_after_burn_in, ]
	plot_filename = paste(PLOTS_DIR, "/rmse_comp_", data_set, "_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, "_alpha_", alpha, "_beta_", beta, ".pdf", sep = "")
	pdf(file = plot_filename)
	boxplot(all_results$A_BART_rmse, all_results$R_BART_rmse, all_results$RF_rmse, 
		names = c("my BART", "Rob's BART", "RF"),
		horizontal = TRUE,
		main = paste("RMSE comparison for ", data_set, ", m = ", num_trees, ", N_B = ", num_burn_in, ", N_G = ", num_iterations_after_burn_in, " alpha = ", alpha, " beta = ", beta, sep = ""),
		xlab = paste("RMSE's (n = ", run_model_N_times, " simulations)", sep = ""))
	dev.off()	
}


create_avg_sim_results_and_save_as_csv = function(){
	for (alpha in alphas_of_interest){
		for (beta in betas_of_interest){	
			for (num_burn_in in num_burn_ins_of_interest){
				for (num_iterations_after_burn_in in num_iterations_after_burn_ins_of_interest){
					for (num_trees in num_trees_of_interest){				
						for (data_set in c(real_regression_data_sets, simulated_data_sets)){
							all_results = simulation_results_pretty[simulation_results_pretty$data_model == data_set & simulation_results_pretty$m == num_trees & simulation_results_pretty$N_B == num_burn_in & simulation_results_pretty$N_G == num_iterations_after_burn_in, ]
							num_a_bart_beats_r_bart = all_results$A_BART_rmse > all_results$R_BART_rmse
							pval_sign_test = pbinom(num_a_bart_beats_r_bart, run_model_N_times, 0.5) #we assume we have an equal chance of beating each other
							new_simul_row = c(
								data_set, 
								num_trees, 
								num_burn_in, 
								num_iterations_after_burn_in,
								alpha,
								beta,
								round(mean(all_results$A_BART_rmse), 1),
								round(sd(all_results$A_BART_rmse), 2),
								round(mean(all_results$R_BART_rmse), 1),
								round(sd(all_results$R_BART_rmse), 2),
								round(pval_sign_test, 3),
								round(mean(all_results$RF_rmse), 1),
								round(sd(all_results$RF_rmse), 2)			
							)
							avg_simulation_results = rbind(avg_simulation_results, new_simul_row)					
						}
					}
				}
			}
		}
	}
	assign("avg_simulation_results", avg_simulation_results, .GlobalEnv)
	#make it pretty right away
	#now update simulation results object
	rownames(avg_simulation_results) = NULL
	avg_simulation_results = as.data.frame(avg_simulation_results)
	for (j in 2 : 11){
		avg_simulation_results[, j] = as.numeric(as.character(avg_simulation_results[, j]))
	}
	#write it to file
	write.csv(avg_simulation_results, paste(PLOTS_DIR, "/", "avg_simulation_results.csv", sep = ""), row.names = FALSE)	
	assign("avg_simulation_results_pretty", avg_simulation_results, .GlobalEnv)
}

data_title = "simple_tree_structure_sigsq_half"
training_data = simulate_data_from_simulation_name(data_title)
test_data = simulate_data_from_simulation_name(data_title)

num_trees = 1
num_burn_in = 2000
num_iterations_after_burn_in = 2000
alpha = 0.95
beta = -2

run_bart_model_and_save_diags_and_results = function(training_data, test_data, data_title, num_trees, num_burn_in, num_iterations_after_burn_in, alpha, beta){
	save_plot = TRUE
	extra_text = paste("on model \"", gsub("_", " ", data_title), "\" m = ", num_trees, " n_B = ", num_burn_in, ", n_G_a = ", 
			num_iterations_after_burn_in, " ", expression(alpha), " = ", alpha,  " ", expression(beta), " = ", beta, sep = "")
	
	#generate the bart model
	bart_machine = bart_model(training_data, 
		num_trees = num_trees, 
		num_burn_in = num_burn_in, 
		alpha = alpha, 
		beta = beta,
		print_tree_illustrations = PRINT_TREE_ILLUS,
		debug_log = JAVA_LOG,
		num_iterations_after_burn_in = num_iterations_after_burn_in)

	ensure_bart_is_done_in_java(bart_machine$java_bart_machine)
	
	#now use the bart model to predict y_hat's for the test data
	a_bart_predictions = predict_and_calc_ppis(bart_machine, test_data)
	#diagnose how good the y_hat's from the bart model are
	plot_y_vs_yhat(a_bart_predictions, extra_text = extra_text, data_title = data_title, save_plot = save_plot, bart_machine = bart_machine)
	
	#now see how Rob's algorithm does
	r_bart_predictions = run_bayes_tree_bart_impl_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text, data_title = data_title, save_plot = save_plot, bart_machine = bart_machine)
	
	#now see how good random forests and CART does in comparison
	rf_predictions = run_random_forests_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text, data_title = data_title, save_plot = save_plot, bart_machine = bart_machine)
	cart_predictions = run_cart_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text, data_title = data_title, save_plot = save_plot, bart_machine = bart_machine)
	
	#do some plots and histograms to diagnose convergence
	plot_sigsqs_convergence_diagnostics(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot)
	hist_sigsqs(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot)
	plot_tree_liks_convergence_diagnostics(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot)
	hist_tree_liks(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot)
	plot_tree_num_nodes(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot)	
	plot_tree_depths(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot)
	for (t in 1 : num_trees){
		plot_all_mu_values_for_tree(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot, t)
		for (b in 1 : maximum_nodes_over_all_trees(bart_machine)){
			hist_mu_values_by_tree_and_leaf_after_burn_in(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot, t, b)
		}
	}
	
	new_simul_row = c(
		data_title, 
		num_trees, 
		num_burn_in, 
		num_iterations_after_burn_in,
		alpha,
		beta,			
		round(a_bart_predictions$L1_err, 0),
		round(a_bart_predictions$L2_err, 0),
		round(a_bart_predictions$rmse, 2),
		round(r_bart_predictions$L1_err, 0),
		round(r_bart_predictions$L2_err, 0),
		round(r_bart_predictions$rmse, 2),	
		round(rf_predictions$L1_err, 0),
		round(rf_predictions$L2_err, 0),
		round(rf_predictions$rmse, 2),
		round(cart_predictions$L1_err, 0),
		round(cart_predictions$L2_err, 0),
		round(cart_predictions$rmse, 2)		
	)
#		print(new_simul_row)
	simulation_results = rbind(simulation_results, new_simul_row)	
#		print(paste("simulation results updated n =", nrow(simulation_results), " p =", ncol(simulation_results), " class =", class(simulation_results), " class =", class(new_simul_row)))
	assign("simulation_results", simulation_results, .GlobalEnv)
#	}, finally = function(){})

}