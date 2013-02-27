#algorithm defaults
num_trees = 200
num_burn_in = 500
num_iterations_after_burn_in = 1000
alpha = 0.95
beta = 2

pct_train = 5 / 6 #same as BART paper

run_model_N_times = 1

real_regression_data_sets = c(
	"r_boston",
	"r_forestfires",
	"r_cars",
	"r_concretedata",
	"r_wine_white",
	"r_wine_red",
	"r_triazine",
	"r_pole",
	"r_ozone",
	"r_cpu",
	"r_compactiv",
	"r_baseballsalary",
	"r_ankara",
	"r_abalone",
	"r_german_credit",
	"r_bbb"
)
simulated_data_sets = c(		
	"univariate_linear",
	"bivariate_linear",
	"friedman", #p=10
	"friedman_p_100",
	"friedman_p_1000",
	"simple_tree_structure_sigsq_3"
)

simulation_results_cols = c(
	"model", 
	"A_BART_rmse",
	"A_BART_CV_rmse",
	"R_BART_rmse",
	"DynaTree_rmse",
	"RF_CV_rmse",	
	"Boosting_CV_rmse",
	"OLS_rmse",
	"Lasso_CV_rmse",
	"Ridge_CV_rmse",	
	"A_BART_L1",
	"A_BART_CV_L1",
	"R_BART_L1",
	"DynaTree_L1",
	"RF_CV_L1",
	"Boosting_CV_L1",
	"Lasso_CV_L1",
	"OLS_L1",
	"Ridge_CV_L1",	
	"A_BART_rmse_train",
	"A_BART_CV_rmse_train",
	"R_BART_rmse_train",
	"DynaTree_rmse_train",
	"RF_CV_rmse_train",
	"Boosting_CV_rmse_train",
	"OLS_rmse_train",
	"Lasso_CV_rmse_train",
	"Ridge_CV_rmse_train",
	"A_BART_tot_var_count",
	"R_BART_tot_var_count",		
	"A_sigsq_post_mean",
	"R_sigsq_post_mean"
)

