#source("r_scripts/bart_bakeoff.R")

directory_where_code_is = getwd() #usually we're on a linux box and we'll just navigate manually to the directory
#if we're on windows, then we're on the dev box, so use a prespecified directory
if (.Platform$OS.type == "windows"){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
}
setwd(directory_where_code_is)

source("r_scripts/bart_package.R")
source("r_scripts/create_simulated_models.R")
graphics.off()

real_regression_data_sets = c(
	"r_boston", 
	"r_forestfires", 
	"r_concretedata"
)

#simulation constants... these can be modulated based on what you want to run
num_trees_of_interest = c(
#	100, 
#	75, 
#	50, 
#	20, 
#	10, 
#	5, 
#	2, 
	1
)

#run_bakeoff()

num_burn_ins_of_interest = c(
#	2000
#	1000,
	500, 
	200,
	100
)
num_iterations_after_burn_ins_of_interest = c(
	2000,
	1000
#	500, 
#	200,
#	100
)

simulation_results = matrix(NA, nrow = 0, ncol = 16)
colnames(simulation_results) = c(
	"data_model", 
	"m",
	"N_B", 
	"N_G", 
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

run_bakeoff = function(){
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
					run_bart_model_and_save_diags_and_results(training_data, test_data, real_regression_data_set, num_trees, num_burn_in, num_iterations_after_burn_in)
				}
				
				for (simulated_data_model_name in simulated_data_model_names){
					training_data = simulate_data_from_simulation_name(simulated_data_model_name)
					test_data = simulate_data_from_simulation_name(simulated_data_model_name)
					
					run_bart_model_and_save_diags_and_results(training_data, test_data, simulated_data_model_name, num_trees, num_burn_in, num_iterations_after_burn_in)				
				}			
			}
		}
	}
}

run_bart_model_and_save_diags_and_results = function(training_data, test_data, data_title, num_trees, num_burn_in, num_iterations_after_burn_in){
	cat(paste("model \"", data_title, "\", m = ", num_trees, ", n_B = ", num_burn_in, ", n_G_a = ", num_iterations_after_burn_in, "\n", sep = ""))
	
	extra_text = paste("on model \"", gsub("_", " ", data_title), "\" m = ", num_trees, " n_B = ", num_burn_in, ", n_G_a = ", num_iterations_after_burn_in, sep = "")
	
	#generate the bart model
	bart_machine = bart_model(training_data, 
		num_trees = num_trees, 
		num_burn_in = num_burn_in, 
#		print_tree_illustrations = TRUE,
#		debug_log = TRUE,
		num_iterations_after_burn_in = num_iterations_after_burn_in)
	
	#do some plots to diagnose convergence
	plot_sigsqs_convergence_diagnostics(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = TRUE)
	plot_tree_liks_convergence_diagnostics(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = TRUE)
	plot_tree_num_nodes(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = TRUE)
	plot_tree_depths(bart_machine, extra_text = extra_text, data_title = data_title, save_plot = TRUE)
	get_root_splits_of_trees(bart_machine, data_title = data_title, save_as_csv = TRUE)

	#now use the bart model to predict y_hat's for the test data
	a_bart_predictions = predict_and_calc_ppis(bart_machine, test_data)
	#diagnose how good the y_hat's from the bart model are
	plot_y_vs_yhat(a_bart_predictions, extra_text = extra_text, data_title = data_title, save_plot = TRUE, bart_machine = bart_machine)
	
	#now see how Rob's algorithm does
	r_bart_predictions = run_bayes_tree_bart_impl_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text, data_title = data_title, save_plot = TRUE, bart_machine = bart_machine)
	
	#now see how good random forests and CART does in comparison
	rf_predictions = run_random_forests_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text, data_title = data_title, save_plot = TRUE, bart_machine = bart_machine)
	cart_predictions = run_cart_and_plot_y_vs_yhat(training_data, test_data, extra_text = extra_text, data_title = data_title, save_plot = TRUE, bart_machine = bart_machine)
	
	new_simul_row = c(
		data_title, 
		num_trees, 
		num_burn_in, 
		num_iterations_after_burn_in,
		a_bart_predictions$L1_err,
		a_bart_predictions$L2_err,
		a_bart_predictions$rmse,
		r_bart_predictions$L1_err,
		r_bart_predictions$L2_err,
		r_bart_predictions$rmse,	
		rf_predictions$L1_err,
		rf_predictions$L2_err,
		rf_predictions$rmse,
		cart_predictions$L1_err,
		cart_predictions$L2_err,
		cart_predictions$rmse		
	)
	simulation_results = rbind(simulation_results, new_simul_row)
}