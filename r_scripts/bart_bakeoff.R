#working directory and libraries
directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
setwd(directory_where_code_is)
PLOTS_DIR = "output_plots"
library(randomForest)
source("r_scripts//bart_package.R")
source("r_scripts//create_simulated_models.R")
graphics.off()


run_bart_model_and_save_diags_and_results = function(training_data, test_data, data_title, num_trees, num_burn_in, num_iterations_after_burn_in){
	extra_text = paste("on model \"", gsub("_", " ", data_title), "\" m = ", num_trees, " n_B = ", num_burn_in, ", n_G_a = ", num_iterations_after_burn_in, sep = "")
	
	#generate the bart model
	model = bart_model(training_data, num_trees = num_trees, num_burn_in = num_burn_in, num_iterations_after_burn_in = num_iterations_after_burn_in)
	
	#do some plots to diagnose convergence
	plot_sigsqs_convergence_diagnostics(model, extra_text = extra_text)
	savePlot(paste(PLOTS_DIR, "//", data_title, "_sigsqs_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, ".pdf", sep = ""), "pdf")
	plot_tree_liks_convergence_diagnostics(model, extra_text = extra_text)
	savePlot(paste(PLOTS_DIR, "//", data_title, "_treeliks_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, ".pdf", sep = ""), "pdf")
	
	#now use the bart model to predict y_hat's for the test data
	predictions = predict_and_calc_ppis(model, test_data)
	#diagnose how good the y_hat's from the bart model are
	plot_y_vs_yhat(predictions, extra_text = extra_text)
	savePlot(paste(PLOTS_DIR, "//", data_title, "_yvyhat_bart_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, ".pdf", sep = ""), "pdf")
	
	#now see how good random forests does in comparison
	run_random_forests_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text)
	savePlot(paste(PLOTS_DIR, "//", data_title, "_yvyhat_rf.pdf", sep = ""), "pdf")	
}

real_regression_data_sets = c("r_boston", "r_forestfires", "r_concretedata")

#some constants for now
num_trees_of_interest = c(1, 2, 5, 10, 20, 50, 75, 100)
num_burn_ins_of_interest = c(50, 100, 200, 500, 1000, 2000)
num_iterations_after_burn_ins_of_interest = c(50, 100, 200, 500, 1000, 2000)

for (num_burn_in in num_burn_ins_of_interest){
	for (num_iterations_after_burn_in in num_iterations_after_burn_ins_of_interest){
		for (num_trees in num_trees_of_interest){
			
			for (real_regression_data_set in real_regression_data_sets){
				cat(paste("model \"", real_regression_data_set, "\", m = ", num_trees, ", n_B = ", num_burn_in, ", n_G_a = ", num_iterations_after_burn_in, "\n", sep = ""))
				
				raw_data = read.csv(paste("datasets//", real_regression_data_set, ".csv", sep = ""))
				training_data = raw_data[seq(from = 1, to = nrow(raw_data), by = 2), ] #all odd rows
				test_data = raw_data[seq(from = 2, to = nrow(raw_data), by = 2), ] #all even rows
				
				run_bart_model_and_save_diags_and_results(training_data, test_data, real_regression_data_set, num_trees, num_burn_in, num_iterations_after_burn_in)
			}
			
			for (simulated_data_model_name in simulated_data_model_names){
				cat(paste("model \"", simulated_data_model_name, "\" m = ", num_trees, " n_B = ", num_burn_in, ", n_G_a = ", num_iterations_after_burn_in, "\n", sep = ""))
				
				training_data = simulate_data_from_simulation_name(simulated_data_model_name)
				test_data = simulate_data_from_simulation_name(simulated_data_model_name)
				
				run_bart_model_and_save_diags_and_results(training_data, test_data, simulated_data_model_name, num_trees, num_burn_in, num_iterations_after_burn_in)				
			}			
		}
	}
}
