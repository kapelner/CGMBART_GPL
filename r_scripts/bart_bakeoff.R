

directory_where_code_is = getwd() #usually we're on a linux box and we'll just navigate manually to the directory
#if we're on windows, then we're on the dev box, so use a prespecified directory
if (.Platform$OS.type == "windows"){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
}
setwd(directory_where_code_is)

source("r_scripts/bart_package.R")
source("r_scripts/create_simulated_models.R")
graphics.off()

source("r_scripts/bart_bakeoff_params.R")

total_num_runs = (length(real_regression_data_sets) + length(simulated_data_sets)) * 
		length(num_trees_of_interest) * length(num_burn_ins_of_interest) * 
		length(num_iterations_after_burn_ins_of_interest) * length(alphas_of_interest) * 
		length(betas_of_interest) * run_model_N_times

num_trees = 1
num_burn_in = 2000
num_iterations_after_burn_in = 2000
alpha = 0.95
beta = 2

simulation_results_cols = c(
		"data_model", 
		"m",
		"N_B", 
		"N_G", 
		"alpha", 
		"beta",
		"A_BART_L1", 
		"R_BART_L1", 
		"A_BART_L2", 
		"R_BART_L2", 
		"A_BART_rmse",
		"R_BART_rmse",
		"A_sigsq_post_mean",
		"R_sigsq_post_mean",
		"A_BART_rmse_train",
		"R_BART_rmse_train",
		"A_BART_tot_var_count",
		"R_BART_tot_var_count",
		"RF_L1",
		"RF_L2",
		"RF_rmse",
		"CART_L1",
		"CART_L2",
		"CART_rmse",
		"A_BART_runtime",
		"R_BART_runtime",
		"RF_runtime",
		"CART_runtime"
)
simulation_results = matrix(NA, nrow = 0, ncol = length(simulation_results_cols))
colnames(simulation_results) = simulation_results_cols

sigsqs_log = matrix(NA, nrow = 0, ncol = 4 + num_burn_in + num_iterations_after_burn_in)


run_bart_bakeoff = function(){
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
							for (duplicate_run in 1 : run_model_N_times){
								cat(paste("model ", real_regression_data_set, "m =", num_trees, "dup = ", duplicate_run, "\n\n"))
								current_run = current_run + 1
								append_to_log(paste("starting model ", current_run, "\\", total_num_runs, "  \"", real_regression_data_set, "\", m = ", num_trees, ", n_B = ", num_burn_in, ", n_G_a = ", num_iterations_after_burn_in, " alpha = ", alpha, " beta = ", beta, sep = ""))
								run_bart_model_and_save_diags_and_results(training_data, test_data, real_regression_data_set, num_trees, num_burn_in, num_iterations_after_burn_in, alpha, beta, duplicate_run)
							}
						}
		
						for (simulated_data_set in simulated_data_sets){
							training_data = simulate_data_from_simulation_name(simulated_data_set)
							test_data = simulate_data_from_simulation_name(simulated_data_set)							
							for (duplicate_run in 1 : run_model_N_times){
								cat(paste("model ", simulated_data_set, "m =", num_trees, "dup = ", duplicate_run, "\n\n"))
								current_run = current_run + 1
								append_to_log(paste("starting model ", current_run, "\\", total_num_runs, "  \"", simulated_data_set, "\", m = ", num_trees, ", n_B = ", num_burn_in, ", n_G_a = ", num_iterations_after_burn_in, " alpha = ", alpha, " beta = ", beta, sep = ""))
								run_bart_model_and_save_diags_and_results(training_data, test_data, simulated_data_set, num_trees, num_burn_in, num_iterations_after_burn_in, alpha, beta, duplicate_run)
							}
						}
					}
				}
			}
		}
	}
	draw_boxplots_of_sim_results()
	calculate_cochran_global_pval()
}

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


run_bart_model_and_save_diags_and_results = function(training_data, test_data, data_title, num_trees, num_burn_in, num_iterations_after_burn_in, alpha, beta, duplicate_run){
	save_plot = TRUE
	extra_text = paste("on model \"", gsub("_", " ", data_title), "\" m = ", num_trees, " n_B = ", num_burn_in, ", n_G_a = ", 
			num_iterations_after_burn_in, " ", expression(alpha), " = ", alpha,  " ", expression(beta), " = ", beta, sep = "")
	
	#generate the bart model
	time_started = Sys.time()
	bart_machine = build_bart_machine(training_data, 
		num_trees = num_trees, 
		num_burn_in = num_burn_in, 
		num_iterations_after_burn_in = num_iterations_after_burn_in,
		alpha = alpha, 
		beta = beta,
		print_tree_illustrations = PRINT_TREE_ILLUS,
		debug_log = JAVA_LOG,
		unique_name = paste(data_title, "_m_", num_trees, "_run_", formatC(duplicate_run, width = 2, format = "d", flag = "0")),
		run_in_sample = TRUE,
		use_heteroskedasticity = FALSE,
		num_cores = num_cores)
	time_finished = Sys.time()
	print(paste("A BART run time:", time_finished - time_started))
	
	assign("bart_machine", bart_machine, .GlobalEnv)
	append_to_log("built")
	
	#now use the bart model to predict y_hat's for the test data
	a_bart_predictions_obj = predict_and_calc_ppis(bart_machine, test_data, num_cores = num_cores)
	#diagnose how good the y_hat's from the bart model are
#	plot_y_vs_yhat_a_bart(a_bart_predictions_obj, extra_text = extra_text, data_title = data_title, save_plot = save_plot, bart_machine = bart_machine)
	
	#now see how Rob's algorithm does
	r_bart_predictions_obj = run_bayes_tree_bart_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text, data_title = data_title, save_plot = save_plot, bart_machine = bart_machine)
	
	#now see how good random forests and CART does in comparison
	rf_predictions_obj = run_random_forests_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text, data_title = data_title, save_plot = save_plot, bart_machine = bart_machine)
	cart_predictions_obj = run_cart_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text, data_title = data_title, save_plot = save_plot, bart_machine = bart_machine)
	
	#do some plots and histograms to diagnose convergence
#	plot_sigsqs_convergence_diagnostics_hetero(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot)
#	a_bart_sigsqs = hist_sigsqs(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = save_plot)
	a_bart_sigsqs = c(0)
#	print(paste("ABART runtime", time_finished - time_started))
#	print(paste("RBART runtime", r_bart_predictions_obj$runtime))
#	print(paste("ABART runtime", rf_predictions_obj$runtime))
#	print(paste("ABART runtime", cart_predictions_obj$runtime))
	new_simul_row = c(
		data_title, 
		num_trees, 
		num_burn_in, 
		num_iterations_after_burn_in,
		alpha,
		beta,
		round(a_bart_predictions_obj$L1_err, 0),
		round(r_bart_predictions_obj$L1_err, 0),
		round(a_bart_predictions_obj$L2_err, 0),
		round(r_bart_predictions_obj$L2_err, 0),
		round(a_bart_predictions_obj$rmse, 2),
		round(r_bart_predictions_obj$rmse, 2),	
		round(mean(a_bart_sigsqs), 3),
		round(mean(r_bart_predictions_obj$sigsqs), 3),
		round(bart_machine$rmse_train, 2),
		round(r_bart_predictions_obj$rmse_train, 2),
		round(sum(bart_machine$avg_num_splits_by_vars), 1),
		round(sum(r_bart_predictions_obj$avg_num_splits_by_vars), 1),
		round(rf_predictions_obj$L1_err, 0),
		round(rf_predictions_obj$L2_err, 0),
		round(rf_predictions_obj$rmse, 2),
		round(cart_predictions_obj$L1_err, 0),
		round(cart_predictions_obj$L2_err, 0),
		round(cart_predictions_obj$rmse, 2),
		#now do runtimes
		round(time_finished - time_started, 2),
		round(r_bart_predictions_obj$runtime, 2),
		round(rf_predictions_obj$runtime, 2),
		round(cart_predictions_obj$runtime, 2)
	)
	simulation_results = rbind(simulation_results, new_simul_row)	
	assign("simulation_results", simulation_results, .GlobalEnv)
	#now prettify and iteratively save so we can cut at any time
	prettify_simulation_results_and_save_as_csv()
	create_avg_sim_results_and_save_as_csv()
	"A BART DONE"
}

if (TRUE){
	run_bart_bakeoff()
}