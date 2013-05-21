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


########## CRAZY MODEL
n_crazy = 1000
p_crazy = 3
prop_missing = 0.1
offset_missing = 5
sigma_e = 1

graphics.off()

Xy = generate_crazy_model(n_crazy, p_crazy, prop_missing, offset_missing, sigma_e)
hist(Xy[, 4], br = 50, main = "distribution of response")
bart_machine = build_bart_machine(Xy = Xy, use_missing_data = TRUE, num_burn_in = 5000)
plot_y_vs_yhat(bart_machine)
windows()
plot_sigsqs_convergence_diagnostics(bart_machine)
bart_machine
check_bart_error_assumptions(bart_machine)
investigate_var_importance(bart_machine)
interaction_investigator(bart_machine, num_replicates_for_avg = 20)

###now do some predictions

###make sure it works...
xnew = as.data.frame(t(as.matrix(c(0, 0, 0))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 0
plot_hist_of_posterior(pred, 0)

xnew = as.data.frame(t(as.matrix(c(1, 0, 0))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 0
plot_hist_of_posterior(pred, 0)

xnew = as.data.frame(t(as.matrix(c(0, 1, 0))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 0
plot_hist_of_posterior(pred, 2)

xnew = as.data.frame(t(as.matrix(c(0, 1, 0))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 0
plot_hist_of_posterior(pred, 2)

xnew = as.data.frame(t(as.matrix(c(1,0,1))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 0
plot_hist_of_posterior(pred, 1)

xnew = as.data.frame(t(as.matrix(c(1, 1, 1))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 0
plot_hist_of_posterior(pred, 4)

xnew = as.data.frame(t(as.matrix(c(NA, 0, 0))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 0
plot_hist_of_posterior(pred, -0.333)

xnew = as.data.frame(t(as.matrix(c(0, NA, 0))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 0
plot_hist_of_posterior(pred, 0.8333)

xnew = as.data.frame(t(as.matrix(c(0, 0, NA))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 3
plot_hist_of_posterior(pred, offset_missing)

xnew = as.data.frame(t(as.matrix(c(NA, NA, 0))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 0
plot_hist_of_posterior(pred, 0.5)

xnew = as.data.frame(t(as.matrix(c(NA, NA, NA))))
pred = bart_machine_predict(bart_machine, xnew) #E[Y] = 3
plot_hist_of_posterior(pred, offset_missing + 0.5)

setwd("C:/Users/Kapelner/Desktop/Dropbox/BSTA_799/final_presentation/images")
