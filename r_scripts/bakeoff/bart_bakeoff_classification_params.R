#algorithm defaults
num_trees = 200
num_burn_in = 500
num_iterations_after_burn_in = 1000
alpha = 0.95
beta = 2

pct_train = 5 / 6 #same as BART paper

run_model_N_times = 10

real_classification_data_sets = c(
	
)
simulated_data_sets = c(		
	
)

simulation_results_cols = c(
	"model", 
	"A_BART_mis",
	"R_BART_mis",
	"RF_CV_mis",
	"A_BART_CV_mis"
)

