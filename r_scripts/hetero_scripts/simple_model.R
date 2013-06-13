LAST_NAME = "kapelner"
NOT_ON_GRID = length(grep("wharton.upenn.edu", Sys.getenv(c("HOSTNAME")))) == 0

if (NOT_ON_GRID){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
} else {
	directory_where_code_is = getwd()
}
setwd(directory_where_code_is)

source("r_scripts/bart_package_inits.R")
source("r_scripts/bart_package_builders.R")
source("r_scripts/bart_package_predicts.R")
source("r_scripts/bart_package_data_preprocessing.R")
source("r_scripts/bart_package_plots.R")
source("r_scripts/bart_package_variable_selection.R")
source("r_scripts/bart_package_f_tests.R")
source("r_scripts/bart_package_summaries.R")
source("r_scripts/missing_data/sims_functions.R")



n = 500
x1 = runif(n)
x2 = runif(n)
sigma = sqrt(exp(2 * x1 + 2 * x2))

y = 3 * x1 + 3 * x2 + rnorm(n, 0, sigma)

Xy = data.frame(x1, x2, y)

lm_mod = lm(y ~ ., Xy)
mse = var(summary(lm_mod)$residuals)



### test BART on it
set_bart_machine_num_cores(4)
bart_machine = build_bart_machine(Xy = Xy, use_heteroskedasticity = TRUE, num_trees = 200)
plot_y_vs_yhat(bart_machine)

