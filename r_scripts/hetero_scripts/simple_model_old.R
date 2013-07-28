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



n = 1000
x1 = runif(n)
sigma = sqrt(exp(0 + 0.5 * x1))
plot(x1, sigma)

y = 10 * x1 + rnorm(n, 0, sigma)
plot(x1, y)



Xy = data.frame(x1, y)
Xytrain = Xy[1 : 500, ]
Xytest = Xy[501 : 1000, ]
mean(sigma)
#
write.csv(Xy, "datasets/r_super_simple_hetero_model.csv", row.names = FALSE)
#Xy = read.csv("datasets/r_super_simple_hetero_model.csv")

#lm_mod = lm(y ~ ., Xy)
#plot(Xy[, 1], Xy[, 2])
#plot(lm_mod)
#mse = var(summary(lm_mod)$residuals)



### test BART on it
set_bart_machine_num_cores(4)
bart_machine = build_bart_machine(Xy = Xytrain, num_trees = 200, mh_prob_steps = c(0.5, 0.5, 0))
bart_machine
new_data = as.matrix(Xytest[, 1])
colnames(new_data)[1] = "x1"
#plot_y_vs_yhat(bart_machine, new_data, Xytest[, 2], ppis = TRUE)
obj = bart_predict_for_test_data(bart_machine, new_data, Xytest[, 2])
obj$rmse




new_data = as.matrix(seq(-0.3, 1.3, 0.005))
colnames(new_data)[1] = "x1"
y_hat = predict(bart_machine, new_data)
plot(new_data, y_hat, ylim = c(-1, 11), main = "vanilla")
#y_hat

windows()


bart_machine_hetero = build_bart_machine(Xy = Xytrain, num_trees = 200, mh_prob_steps = c(0.5, 0.5, 0), use_linear_heteroskedasticity_model = TRUE)
bart_machine_hetero
plot_y_vs_yhat(bart_machine_hetero, ppis = TRUE)
new_data = as.matrix(Xytest[, 1])
colnames(new_data)[1] = "x1"
obj_hetero = bart_predict_for_test_data(bart_machine_hetero, new_data, Xytest[, 2])
obj_hetero$rmse


new_data = as.matrix(seq(-0.3, 1.3, 0.005))
colnames(new_data)[1] = "x1"
y_hat = predict(bart_machine_hetero, new_data)
plot(new_data, y_hat, ylim = c(-1, 11), main = "hetero")
#y_hat


#windows()
plot_tree_depths(bart_machine)
#plot_tree_num_nodes(bart_machine)
#plot_mh_acceptance_reject(bart_machine)



