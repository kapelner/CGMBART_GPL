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
source("r_scripts/bart_package_hetero_tests.R")
source("r_scripts/bart_package_summaries.R")
source("r_scripts/bart_package_validation.R")




###########same!!!!!!

n = 1000
p = 1

X = cbind(matrix(runif(n * p, 0, 3), ncol = p))

#generate mean model
beta = seq(-1, 1, length.out = p)

#generate heteroskedastic model
gamma_0 = -5
gamma = c(3)
sigma = sqrt(exp(gamma_0 + X %*% gamma))
y = as.numeric(X %*% beta + rnorm(n, 0, sigma))

Xy = data.frame(X, y)
par(mfrow = c(1, p + 1))
for (j in 1 : p){
	plot(X[, j], y, xlab = paste0("x_", j))
}
plot(y, sigma)


kf_homo = k_fold_cv(X, y, k_folds = 2, num_burn_in = 1000)
kf_homo
kf_hetero = k_fold_cv(X, y, use_linear_heteroskedasticity_model = TRUE, k_folds = 2, num_burn_in = 1000)
kf_hetero


cat((kf_homo$rmse - kf_hetero$rmse) / kf_hetero$rmse * 100, "% better ... yeah buddy\n", sep = "")

bart_machine_hetero = build_bart_machine(as.data.frame(X), y,
		num_trees = 200,
		num_burn_in = 500,
		num_iterations_after_burn_in = 1000,
		use_linear_heteroskedasticity_model = TRUE)
bart_machine_hetero
gg = getGammas(bart_machine_hetero)
ggab = gg[(bart_machine_hetero$num_burn_in + 1) : nrow(gg), ]
paste(colMeans(ggab), "+-", apply(ggab, 2, sd))

# this model was somewhat hetero
# observations: does just 2% > homo, does not find model

############################################

#generate heteroskedastic model
gamma_0 = -10
gamma = c(5)
sigma = sqrt(exp(gamma_0 + X %*% gamma))
y = as.numeric(X %*% beta + rnorm(n, 0, sigma))

Xy = data.frame(X, y)
par(mfrow = c(1, p + 1))
for (j in 1 : p){
	plot(X[, j], y, xlab = paste0("x_", j))
}
plot(y, sigma)


kf_homo = k_fold_cv(X, y, k_folds = 2)
kf_homo
kf_hetero = k_fold_cv(X, y, use_linear_heteroskedasticity_model = TRUE, k_folds = 2)
kf_hetero


cat((kf_homo$rmse - kf_hetero$rmse) / kf_hetero$rmse * 100, "% better ... yeah buddy\n", sep = "")


bart_machine_hetero = build_bart_machine(as.data.frame(X), y, num_trees = 200, use_linear_heteroskedasticity_model = TRUE)
bart_machine_hetero
gg = getGammas(bart_machine_hetero)
ggab = gg[(bart_machine_hetero$num_burn_in + 1) : nrow(gg), ]
paste(colMeans(ggab), "+-", apply(ggab, 2, sd))

# this model was very hetero
# observations: does just 4.5% > homo, does not find model


###################################

gamma_0 = 1
gamma = c(0)
sigma = sqrt(exp(gamma_0 + X %*% gamma))
y = as.numeric(X %*% beta + rnorm(n, 0, sigma))

Xy = data.frame(X, y)
par(mfrow = c(1, p + 1))
for (j in 1 : p){
	plot(X[, j], y, xlab = paste0("x_", j))
}
plot(y, sigma)


kf_homo = k_fold_cv(X, y, k_folds = 2)
kf_homo
kf_hetero = k_fold_cv(X, y, use_linear_heteroskedasticity_model = TRUE, k_folds = 2)
kf_hetero


cat((kf_homo$rmse - kf_hetero$rmse) / kf_hetero$rmse * 100, "% better ... yeah buddy\n", sep = "")


bart_machine_hetero = build_bart_machine(as.data.frame(X), y, num_trees = 200, use_linear_heteroskedasticity_model = TRUE)
bart_machine_hetero
gg = getGammas(bart_machine_hetero)
ggab = gg[(bart_machine_hetero$num_burn_in + 1) : nrow(gg), ]
paste(colMeans(ggab), "+-", apply(ggab, 2, sd))

# this model was not hetero
# observations: does just as well on homo and hetero, finds models

###################################

gamma_0 = 1
gamma = c(0)
sigma = sqrt(exp(gamma_0 + X %*% gamma))
y = as.numeric(X %*% beta + rnorm(n, 0, sigma))

Xy = data.frame(X, y)
par(mfrow = c(1, p + 1))
for (j in 1 : p){
	plot(X[, j], y, xlab = paste0("x_", j))
}
plot(y, sigma)


kf_homo = k_fold_cv(X, y, k_folds = 2)
kf_homo
kf_hetero = k_fold_cv(X, y, use_linear_heteroskedasticity_model = TRUE, k_folds = 2)
kf_hetero


cat((kf_homo$rmse - kf_hetero$rmse) / kf_hetero$rmse * 100, "% better ... yeah buddy\n", sep = "")


bart_machine_hetero = build_bart_machine(as.data.frame(X), y, num_trees = 200, use_linear_heteroskedasticity_model = TRUE)
bart_machine_hetero
gg = getGammas(bart_machine_hetero)
ggab = gg[(bart_machine_hetero$num_burn_in + 1) : nrow(gg), ]
paste(colMeans(ggab), "+-", apply(ggab, 2, sd))


# this model was homo
# observations: does just as well on homo and hetero, finds models

###################################

#example from paper

n = 1000

beta_0 = -35
x2 = runif(n, 0, 400)
x3 = runif(n, 10, 23)
x4 = runif(n, 0, 10)
X = cbind(x2, x3, x4)

#generate mean model
beta = c(.35, -1.7, 0)

gamma_0 = -8
gamma = c(0.026, 0, -0.4)
sigma = sqrt(exp(gamma_0 + X %*% gamma))
mean(sigma^2)
y = as.numeric(beta_0 + X %*% beta + rnorm(n, 0, sigma))

Xy = data.frame(X, y)
par(mfrow = c(1, ncol(X) + 1))
for (j in 1 : ncol(X)){
	plot(X[, j], y, xlab = paste0("x_", j))
}
plot(y, sigma)


kf_homo = k_fold_cv(X, y, k_folds = 2)
kf_homo
kf_hetero = k_fold_cv(X, y, use_linear_heteroskedasticity_model = TRUE, k_folds = 2)
kf_hetero


cat((kf_homo$rmse - kf_hetero$rmse) / kf_hetero$rmse * 100, "% better ... yeah buddy\n", sep = "")


bart_machine_hetero = build_bart_machine(as.data.frame(X), y, use_linear_heteroskedasticity_model = TRUE)
bart_machine_hetero
gg = getGammas(bart_machine_hetero)
ggab = gg[(bart_machine_hetero$num_burn_in + 1) : nrow(gg), ]
paste(colMeans(ggab), "+-", apply(ggab, 2, sd))
