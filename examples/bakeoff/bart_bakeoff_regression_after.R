
#draw_boxplots_of_sim_results()
#calculate_cochran_global_pval()

calculate_cochran_global_pval = function(){
	n = nrow(avg_simulation_results)
	chi_sq = sum(-2 * log(as.numeric(avg_simulation_results[, "pval_sign_test"])))
	1 - pchisq(chi_sq, 2 * n)
}


prettify_simulation_results_and_save_as_csv = function(){
	#now update simulation results object
	rownames(simulation_results) = NULL
	simulation_results = as.data.frame(simulation_results)
	for (j in 2 : ncol(simulation_results)){
		simulation_results[, j] = as.numeric(as.character(simulation_results[, j]))
	}
	#assign it to the object
	assign("simulation_results_pretty", simulation_results, .GlobalEnv)
	#write it to file
	write.csv(simulation_results, paste("simulation_results", "/", "simulation_results.csv", sep = ""), row.names = FALSE)
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


avg_simulation_results_cols = c(
		"data_model", 
		"m",
		"N_B", 
		"N_G",
		"alpha", 
		"beta",
		"A_BART_rmse_avg", 
		"R_BART_rmse_avg",
		"RF_rmse_avg",
		"A_BART_rmse_se",		
		"R_BART_rmse_se",
		"A_BART_sigsq",		
		"R_BART_sigsq",	
		"A_BART_rmse_train",		
		"R_BART_rmse_train",
		"A_BART_tot_var_count",		
		"R_BART_tot_var_count",			
		"pval_sign_test",
		"A_BART_runtime",
		"R_BART_runtime",
		"RF_runtime"
)
avg_simulation_results = matrix(NA, nrow = 0, ncol = length(avg_simulation_results_cols))
colnames(avg_simulation_results) = avg_simulation_results_cols

create_avg_sim_results_and_save_as_csv = function(){
	for (alpha in alphas_of_interest){
		for (beta in betas_of_interest){	
			for (num_burn_in in num_burn_ins_of_interest){
				for (num_iterations_after_burn_in in num_iterations_after_burn_ins_of_interest){
					for (num_trees in num_trees_of_interest){				
						for (data_set in c(real_regression_data_sets, simulated_data_sets)){
							all_results_for_run = simulation_results_pretty[simulation_results_pretty$data_model == data_set & simulation_results_pretty$m == num_trees & simulation_results_pretty$N_B == num_burn_in & simulation_results_pretty$N_G == num_iterations_after_burn_in, ]
							num_a_bart_beats_r_bart = sum(all_results_for_run$A_BART_rmse > all_results_for_run$R_BART_rmse)
							pval_sign_test = binom.test(num_a_bart_beats_r_bart, run_model_N_times, 0.5)$p.value
							new_simul_row = c(
									data_set, 
									num_trees, 
									num_burn_in, 
									num_iterations_after_burn_in,
									alpha,
									beta,
									round(mean(all_results_for_run$A_BART_rmse), 2),
									round(mean(all_results_for_run$R_BART_rmse), 2),
									round(mean(all_results_for_run$RF_rmse), 1),
									round(sd(all_results_for_run$A_BART_rmse), 2),								
									round(sd(all_results_for_run$R_BART_rmse), 2),
									round(mean(all_results_for_run$A_sigsq_post_mean), 2),
									round(mean(all_results_for_run$R_sigsq_post_mean), 2),
									round(mean(all_results_for_run$A_BART_rmse_train), 2),
									round(mean(all_results_for_run$R_BART_rmse_train), 2),	
									round(mean(all_results_for_run$A_BART_tot_var_count), 2),
									round(mean(all_results_for_run$R_BART_tot_var_count), 2),								
									round(pval_sign_test, 3),
									round(all_results_for_run$A_BART_runtime, 1),
									round(all_results_for_run$R_BART_runtime, 1),
									round(all_results_for_run$RF_runtime, 1)
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
	for (j in 2 : ncol(avg_simulation_results)){
		avg_simulation_results[, j] = as.numeric(as.character(avg_simulation_results[, j]))
	}
	#write it to file
	write.csv(avg_simulation_results, paste("simulation_results", "/", "avg_simulation_results.csv", sep = ""), row.names = FALSE)	
	assign("avg_simulation_results_pretty", avg_simulation_results, .GlobalEnv)
}

data_title = "simple_tree_structure_sigsq_half"
training_data = simulate_data_from_simulation_name(data_title)
test_data = simulate_data_from_simulation_name(data_title)