LAST_NAME = "kapelner"
NOT_ON_GRID = length(grep("wharton.upenn.edu", Sys.getenv(c("HOSTNAME")))) == 0

if (NOT_ON_GRID){
	directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
} else {
	directory_where_code_is = getwd()
}
setwd(directory_where_code_is)


n = 1000
p = 1

X = cbind(matrix(runif(n * p, 0, 1), ncol = p))

#generate heteroskedastic model
gamma_0 = -5
gamma = c(3)
sigma = sqrt(exp(gamma_0 + X %*% gamma))

#generate mean model
beta = seq(-1, 1, length.out = p)
y = as.numeric(X %*% beta + rnorm(n, 0, sigma))


Xy = data.frame(X, y)
par(mfrow = c(1, p + 1))
for (j in 1 : p){
	plot(X[, j], y, xlab = paste0("x_", j))
}
plot(y, sigma)

X = as.data.frame(X)
write.csv(Xy, "datasets/r_simple_hetero_model_bigger.csv", row.names = FALSE)
#write.csv(Xy, "datasets/r_simple_hetero_model.csv", row.names = FALSE)
#Xy = read.csv("datasets/r_simple_hetero_model.csv")




source("r_scripts/bart_package_inits.R")
source("r_scripts/bart_package_builders.R")
source("r_scripts/bart_package_predicts.R")
source("r_scripts/bart_package_data_preprocessing.R")
source("r_scripts/bart_package_plots.R")
source("r_scripts/bart_package_variable_selection.R")
source("r_scripts/bart_package_f_tests.R")
source("r_scripts/bart_package_hetero_tests.R")
source("r_scripts/bart_package_summaries.R")
source("r_scripts/bart_package_validation.R")


set_bart_machine_num_cores(1)
bart_machine_homo = build_bart_machine(X, y,
		num_trees = 200,
		num_burn_in = 500,
		num_iterations_after_burn_in = 1000)
bart_machine_homo
bart_machine_hetero = build_bart_machine(X, y,
		num_trees = 200,
		num_burn_in = 500,
		num_iterations_after_burn_in = 1000,
		use_linear_heteroskedasticity_model = TRUE)
bart_machine_hetero

plot_y_vs_yhat(bart_machine_homo)
y_hat_homo = predict(bart_machine_homo, X)
plot_y_vs_yhat(bart_machine_hetero)
y_hat_hetero = predict(bart_machine_hetero, X)

graphics.off()
plot(y_hat_homo, y_hat_hetero)
cor(y_hat_homo, y_hat_hetero)

gg = getGammas(bart_machine_hetero)

ggab = gg[(bart_machine_hetero$num_burn_in + 1) : nrow(gg), ]
colMeans(ggab)
apply(ggab, 2, sd)

avg_gammas = colMeans(ggab)
avg_gammas


true_sigsqs = sigma^2

corrs = array(NA, n)
for (i in 1 : n){
	est_sigsqs = exp(as.matrix(X) %*% as.matrix(ggab[i, ]))
	
	corrs[i] = cor(true_sigsqs, est_sigsqs)	
}
hist(corrs, br = 50)


est_sigsqs = exp(as.matrix(X) %*% as.matrix(avg_gammas))

cor(true_sigsqs, as.numeric(est_sigsqs))
plot(true_sigsqs, est_sigsqs)


kf = k_fold_cv(X, y, k_folds = 3)
kf
kf_hetero = k_fold_cv(X, y, use_linear_heteroskedasticity_model = TRUE, k_folds = 3)
kf_hetero














#is there heteroskedasticity?
#bart_machine = build_bart_machine(Xy = Xy)
#test_heteroskedasticity_of_bart_model(bart_machine)

Xytrain = Xy[1 : 500, ]
Xytest = Xy[501 : 1000, ]
#summary(sigma)
mean(sigma)
mean(sigma^2)


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

mod = lm(log(e_i_sqds) ~ x1 + x2 + x3)
summary(mod)
gamma
#log(mse)
#log(sqrt(mse))
#log(mean(sigma^2))

expe_log_chisq_1 = 1.274693


(coef(summary(mod))[1, 1]) + expe_log_chisq_1

plot(x1, mod$fitted)
points(x1, sigma^2, col = "red")

bart_machine_hetero = build_bart_machine(y = e_i_sqds, X = data.frame(x1), run_in_sample = FALSE)

#xs = seq(from = -0.1, to = 1.1, by = 0.001)
xs = cbind(seq(from = -0.1, to = 1.1, by = 0.001), seq(from = -0.1, to = 1.1, by = 0.001))
sigsq_preds = predict(bart_machine_hetero, as.matrix(xs))

plot(xs[, 1], sigsq_preds, xlim = c(-0.1, 1.1))
points(xs[, 1], 2 + 5 * xs[, 1] - 2 * xs[, 2], col = "red")

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
bart_machine_hetero = build_bart_machine(Xy = Xytrain)
bart_machine_hetero
#colnames(new_data)[1] = "x1"
#plot_y_vs_yhat(bart_machine, new_data, Xytest[, 2], ppis = TRUE)
obj = bart_predict_for_test_data(bart_machine_hetero, Xytest[, 1 : 3], Xytest[, 4])
obj$rmse


#test_heteroskedasticity_of_bart_model(bart_machine)

#new_data = cbind(as.matrix(seq(-0.3, 1.3, 0.005)), as.matrix(seq(-0.3, 1.3, 0.005))) 
#
#y_hat = predict(bart_machine, new_data)
#plot(new_data[, 1], y_hat, ylim = c(-1, 11), main = "vanilla")
##y_hat
##
#windows()



set_bart_machine_num_cores(4)
hbart_machine = build_bart_machine(Xy = Xytrain, use_linear_heteroskedasticity_model = TRUE)
hbart_machine

hobj = bart_predict_for_test_data(hbart_machine, Xytest[, 1 : 3], Xytest[, 4])
hobj$rmse


cat((obj$rmse - hobj$rmse) / hobj$rmse * 100, "% better ... yeah buddy\n", sep = "")

par(mfrow = c(1, 1))
plot_y_vs_yhat(bart_machine_hetero, Xytest[, 1 : 3], Xytest[, 4], ppis = TRUE)
windows()
plot_y_vs_yhat(hbart_machine, Xytest[, 1 : 3], Xytest[, 4], ppis = TRUE)


#obj = bart_machine_predict(bart_machine_hetero, as.matrix(c(0.1)))
#hist(as.numeric(obj$y_hat_posterior_samples))
#var(as.numeric(obj$y_hat_posterior_samples))
#obj = bart_machine_predict(bart_machine_hetero, as.matrix(c(0.9)))
#hist(obj$y_hat_posterior_samples)
#var(as.numeric(obj$y_hat_posterior_samples))

windows()
new_data = cbind(as.matrix(seq(-0.3, 1.3, 0.005)), as.matrix(seq(-0.3, 1.3, 0.005))) 

y_hat = predict(bart_machine_hetero, new_data)
plot(new_data[, 1], y_hat, ylim = c(-1, 11), main = "hetero")
#y_hat





library(MASS)
data(Boston)
X = Boston[sample(1 : nrow(Boston), nrow(Boston)), ]
#X = cbind(X, rnorm(nrow(X)))
y = X$medv
X$medv = NULL
#X$chas = as.character(X$chas)
#X$rad = as.factor(X$rad)

#split it into test and training
Xtrain = X[1 : (nrow(X) / 2), ]
ytrain = y[1 : (nrow(X) / 2)]
Xtest = X[(nrow(X) / 2 + 1) : nrow(X), ]
ytest = y[(nrow(X) / 2 + 1) : nrow(X)]

set_bart_machine_num_cores(4)
bart_machine_hetero = build_bart_machine(Xtrain, ytrain, num_burn_in = 500, num_iterations_after_burn_in = 1000)
bart_machine_hetero
#plot_convergence_diagnostics(bart_machine)

plot_y_vs_yhat(bart_machine_hetero, Xtest, ytest, ppis = TRUE)
predict_obj = bart_predict_for_test_data(bart_machine_hetero, Xtest, ytest)
predict_obj$rmse


set_bart_machine_num_cores(4)
hbart_machine = build_bart_machine(Xtrain, ytrain, num_burn_in = 500, num_iterations_after_burn_in = 1000, use_linear_heteroskedasticity_model = TRUE)
hbart_machine

windows(); plot_y_vs_yhat(hbart_machine, Xtest, ytest, ppis = TRUE)
hpredict_obj = bart_predict_for_test_data(hbart_machine, Xtest, ytest)
hpredict_obj$rmse

e_i_sqds = (ytrain - bart_machine_hetero$y_hat)^2
mod = lm(e_i_sqds ~ ., Xtrain)
summary(mod)

library(randomForest)
mod = randomForest(y = ytrain, x = Xtrain)
y_hat = predict(mod, Xtest)
sqrt(sum((ytest - y_hat)^2) / nrow(Xtest))






#######is Boston heteroskedastic??


bart_machine_hetero = build_bart_machine(Boston[, 1 : 13], Boston[, 14], num_burn_in = 500, num_iterations_after_burn_in = 1000)
bart_machine_hetero

test_heteroskedasticity_of_bart_model(bart_machine_hetero)




