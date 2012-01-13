#working directory and libraries
directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
setwd(directory_where_code_is)
library(randomForest)
source("r_scripts//bart_package.R")
source("r_scripts//create_simulated_models.R")
graphics.off()

#some constants for now
num_trees = 50
num_burn_in = 1000
num_iterations_after_burn_in = 1000

for (simulated_data_model_name in simulated_data_model_names){
	
	extra_text = paste("on model \"", gsub("_", " ", simulated_data_model_name), "\" m = ", num_trees, ", n_G_after = ", num_iterations_after_burn_in, sep = "")
	
	#first simulate the training and test data
	training_data = simulate_data_from_simulation_name(simulated_data_model_name)
	test_data = simulate_data_from_simulation_name(simulated_data_model_name)

	#generate the bart model
	model = bart_model(training_data, num_trees = num_trees, num_burn_in = num_burn_in, num_iterations_after_burn_in = num_iterations_after_burn_in)
	
	#do some plots to diagnose convergence
	plot_sigsqs_convergence_diagnostics(model, extra_text = extra_text)
	plot_tree_liks_convergence_diagnostics(model, extra_text = extra_text)
	
	#now use the bart model to predict y_hat's for the test data
	predictions = predict_and_calc_ppis(model, test_data)
	#diagnose how good the y_hat's from the bart model are
	plot_y_vs_yhat(predictions, extra_text = extra_text)
	
	#now see how good random forests does in comparison
	run_random_forests_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text)
}





