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
#x2 = runif(n)
#sigma = sqrt((0.55 + 3.25 * x1 + 0.5 * x2))
sigma = sqrt((0 + 0.5 * x1))
plot(x1, sigma)

y = 10 * x1 + rnorm(n, 0, sigma)
Xy = data.frame(x1, y)
#Xy = data.frame(x1, x2, y)
plot(x1, y)

Xytrain = Xy[1 : 500, ]
Xytest = Xy[501 : 1000, ]
#summary(sigma)
#mean(sigma)
#mean(sigma^2)

write.csv(Xy, "datasets/r_super_simple_hetero_model.csv", row.names = FALSE)
#Xy = read.csv("datasets/r_super_simple_hetero_model.csv")

lm_mod = lm(y ~ ., Xy)
mse = var(summary(lm_mod)$residuals)


summary(lm_mod)
#summary(lm_mod$fitted)
e_i_sqds = summary(lm_mod)$residuals^2


#x1_adj = x1- mean(x1)
#y_adj = log(e_i_sqds)-log(mse)
#mod = lm(y_adj~0+x1_adj)
#summary(mod)
#summary(x1*mod$coef)

mod = lm((e_i_sqds) ~ x1 + x2)
summary(mod)
log(mse)

plot(x1, mod$fitted + mse)
points(x1, log(sigma^2), col = "red")

bart_machine = build_bart_machine(y = e_i_sqds, X = x1, run_in_sample = FALSE)

xs = seq(from = -0.1, to = 1.1, by = 0.001)
sigsq_preds = predict(bart_machine, as.matrix(xs))

plot(xs, sigsq_preds, xlim = c(0, 1))
points(xs, exp(0 + 0.5 * xs), col = "red")

cbind(xs, sigsq_preds)

sigsqs_lm_adj = log(e_i_sqds) + log(mean(e_i_sqds))
mod=lm(sigsqs_lm_adj ~  Xy[, 1])
summary(mod)

plot(x1, sigsqs_lm_adj)
abline(a = 0, b = 0.5, col = "blue")
abline(mod, col = "red")
windows(); hist(sigsqs_lm_adj[x1 < 0.05], br = 50)



#plot(Xy[, 1], Xy[, 2])
##plot(lm_mod)
#mse = var(summary(lm_mod)$residuals)
#log_sigsqs_lm = log(summary(lm_mod)$residuals^2)


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
#
windows()



set_bart_machine_num_cores(4)
bart_machine_hetero = build_bart_machine(Xy = Xytrain, num_trees = 200, mh_prob_steps = c(0.5, 0.5, 0), use_heteroskedasticity = TRUE)
bart_machine_hetero
plot_y_vs_yhat(bart_machine_hetero, ppis = TRUE)
new_data = as.matrix(Xytest[, 1])
colnames(new_data)[1] = "x1"
obj_hetero = bart_predict_for_test_data(bart_machine_hetero, new_data, Xytest[, 2])
obj_hetero$rmse

#obj = bart_machine_predict(bart_machine_hetero, as.matrix(c(0.1)))
#hist(as.numeric(obj$y_hat_posterior_samples))
#var(as.numeric(obj$y_hat_posterior_samples))
#obj = bart_machine_predict(bart_machine_hetero, as.matrix(c(0.9)))
#hist(obj$y_hat_posterior_samples)
#var(as.numeric(obj$y_hat_posterior_samples))


new_data = as.matrix(seq(-0.3, 1.3, 0.005))
colnames(new_data)[1] = "x1"
y_hat = predict(bart_machine_hetero, new_data)
plot(new_data, y_hat, ylim = c(-1, 11), main = "hetero")
#y_hat



