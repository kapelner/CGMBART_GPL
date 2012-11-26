
PRINT_TREE_ILLUS = FALSE
JAVA_LOG = FALSE

num_cores = 15 #a blade is 16 and I want to keep one core for switching and miscellaneous

run_model_N_times = 10
real_regression_data_sets = c(
	"r_boston",
	"r_forestfires"
#	"r_concretedata"
)
simulated_data_sets = c(
	"just_noise_linear",		
	"univariate_linear",
#	"bivariate_linear",
	"friedman",
	"simple_tree_structure_sigsq_hundredth",
#	"simple_tree_structure_sigsq_tenth",
#	"simple_tree_structure_sigsq_half",
	"simple_tree_structure",
#	"simple_tree_structure_sigsq_3",
	"simple_tree_structure_sigsq_5"
#	"simple_tree_structure_sigsq_10"
#	"simple_tree_structure_sigsq_30"
#	"simple_tree_structure_sigsq_100"
)

#nice to have data around for testing... should be overwritten for custom runs...
num_trees_of_interest = c(
	100, 200
)
num_burn_ins_of_interest = c(
	2000
)
num_iterations_after_burn_ins_of_interest = c(
	2000
)
alphas_of_interest = c(0.95)
betas_of_interest = c(2)
