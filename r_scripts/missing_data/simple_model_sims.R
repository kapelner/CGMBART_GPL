
directory_where_code_is = getwd() #usually we're on a linux box and we'll just navigate manually to the directory
#if we're on windows, then we're on the dev box, so use a prespecified directory
if (.Platform$OS.type == "windows"){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
}
setwd(directory_where_code_is)

source("r_scripts/bart_package.R")
source("r_scripts/bart_package_plots.R")
source("r_scripts/bart_package_variable_selection.R")
source("r_scripts/bart_package_f_tests.R")
source("r_scripts/missing_data/sims_functions.R")

Xy = generate_simple_model_with_missingness(500, 0, 20, gamma = 0.1)
Xytest = generate_simple_model_with_missingness(500, 0, 20, gamma = 0.1)

set_bart_machine_num_cores(4)
bart_machine = build_bart_machine(Xy = Xy,
		num_trees = 200,
		num_burn_in = 1000,
		cov_prior_vec = c(1, 10),
		num_iterations_after_burn_in = 1000,
		use_missing_data = TRUE)
bart_machine
#plot_convergence_diagnostics(bart_machine)
#plot_tree_depths(bart_machine)
#bart_machine$training_data_features
#head(bart_machine$model_matrix_training_data)
plot_y_vs_yhat(bart_machine)
plot_y_vs_yhat(bart_machine, ppis = TRUE)
X1 = as.matrix(Xytest[, 1])
colnames(X1) = colnames(Xytest)[1]
windows()
plot_y_vs_yhat(bart_machine, X = X1, y = Xytest[, 2], ppis = TRUE)

x_new = as.matrix(NA)
colnames(x_new) = "X_1"
predict_obj = bart_machine_predict(bart_machine, x_new)
hist(predict_obj$y_hat_posterior_samples, br=100)
predict_obj$y_hat






training_data = generate_simple_model_probit_with_missingness(400, mu_1 = -1, mu_2 = 1, gamma = 0.2)
Xy = training_data$Xy

set_bart_machine_num_cores(4)
bart_machine = build_bart_machine(Xy = Xy,
		num_trees = 200,
		num_burn_in = 1000,
		cov_prior_vec = c(1, 10),
		num_iterations_after_burn_in = 1000,
		use_missing_data = TRUE)
bart_machine
#plot_convergence_diagnostics(bart_machine)
#plot_tree_depths(bart_machine)
#bart_machine$training_data_features
#head(bart_machine$model_matrix_training_data)
#plot_y_vs_yhat(bart_machine)
#plot_y_vs_yhat(bart_machine, ppis = TRUE)
#X1 = as.matrix(Xytest[, 1])
#colnames(X1) = colnames(Xytest)[1]
#windows()
#plot_y_vs_yhat(bart_machine, X = X1, y = Xytest[, 2], ppis = TRUE)

x_new = as.matrix(Xy[, 1])
colnames(x_new) = "X_1"
predict_obj = bart_machine_predict(bart_machine, x_new)
#hist(predict_obj$y_hat_posterior_samples, br=100)
#predict_obj$y_hat

#bart_machine$p_hat_train
y_hat_train = ifelse(predict_obj$y_hat > 0.5, 1, 0)

windows()
plot(training_data$probs, predict_obj$y_hat, xlim = c(0,1), ylim = c(0,1))
abline(a = 0, b = 1)
#cbind(Xy$Y, y_hat_train)













