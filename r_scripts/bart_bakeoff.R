#working directory and libraries
directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
setwd(directory_where_code_is)
library(randomForest)
source("r_scripts//bart_package.R")
source("r_scripts//create_simulated_models.R")
graphics.off()

PROP_OF_DATA_IS_TRAINING = 0.5

#create data and split into training and test
all_data = simu_data_mod__bivariate_linear()
training_data = all_data[1 : floor(nrow(all_data) * PROP_OF_DATA_IS_TRAINING), ]
test_data = all_data[(floor(nrow(all_data) * PROP_OF_DATA_IS_TRAINING) + 1) : nrow(all_data), ]

num_trees = 50
num_burn_in = 50
num_iterations_after_burn_in = 100
debug_log = FALSE

model = bart_model(training_data, 
			num_trees = num_trees, 
			num_burn_in = num_burn_in, 
			num_iterations_after_burn_in = num_iterations_after_burn_in, 
			debug_log = debug_log)
	
plot_sigsqs_convergence_diagnostics(model)
plot_tree_liks_convergence_diagnostics(model)

###### NEED 1) loop over data sets

predictions = predict_and_calc_ppis(model, test_data)
plot_y_vs_yhat(predictions)

l1err_bart = predictions$L1_err
l2err_bart = predictions$L2_err

rf_mod = randomForest(y ~., training_data)
yhat_rf = predict(rf_mod, test_data)

l1err_rf = round(sum(abs(test_data$y - yhat_rf)), 1)
l2err_rf = round(sum((test_data$y - yhat_rf)^2), 1)

windows()
plot(test_data$y, 
	yhat_rf, 
	main = paste("y vs yhat rf model\nL1err = ", l1err_rf, " L2err = ", l2err_rf, sep = ""), 
	xlab = "y", 
	ylab = "y_hat")





