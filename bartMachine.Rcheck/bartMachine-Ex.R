pkgname <- "bartMachine"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
options(pager = "console")
base::assign(".ExTimings", "bartMachine-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('bartMachine')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
cleanEx()
nameEx("bart_machine_get_posterior")
### * bart_machine_get_posterior

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: bart_machine_get_posterior
### Title: Get Full Posterior Distribution
### Aliases: bart_machine_get_posterior
### Keywords: ~kwd1 ~kwd2

### ** Examples

#Regression example

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

#get posterior distribution
posterior = bart_machine_get_posterior(bart_machine, X)
print(posterior$y_hat)

#destroy BART model
destroy_bart_machine(bart_machine)


#Classification example

#get data and only use 2 factors
data(iris)
iris2 = iris[51:150,]
iris2$Species = factor(iris2$Species)

#build BART classification model
bart_machine = build_bart_machine(iris2[ ,1 : 4], iris2$Species)

#get posterior distribution
posterior = bart_machine_get_posterior(bart_machine, iris2[ ,1 : 4])
print(posterior$y_hat)

#destroy BART model
destroy_bart_machine(bart_machine)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("bart_machine_get_posterior", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("bart_machine_num_cores")
### * bart_machine_num_cores

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: bart_machine_num_cores
### Title: Get Number of Cores Used by BART
### Aliases: bart_machine_num_cores
### Keywords: ~kwd1 ~kwd2

### ** Examples

bart_machine_num_cores()



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("bart_machine_num_cores", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("bart_predict_for_test_data")
### * bart_predict_for_test_data

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: bart_predict_for_test_data
### Title: Predict for Test Data with Known Outcomes
### Aliases: bart_predict_for_test_data
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 400 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##split into train and test
train_X = X[1 : 200, ]
test_X = X[201 : 400, ]
train_y = y[1 : 200]
test_y = y[201 : 400]

##build BART regression model
bart_machine = build_bart_machine(train_X, train_y)

#explore performance on test data
oos_perf = bart_predict_for_test_data(bart_machine, test_X, test_y)
print(oos_perf$rmse)

#destroy BART model
destroy_bart_machine(bart_machine)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("bart_predict_for_test_data", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("build_bart_machine")
### * build_bart_machine

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: build_bart_machine
### Title: Build a BART Model
### Aliases: build_bart_machine
### Keywords: ~kwd1 ~kwd2

### ** Examples

##regression example

##generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)
summary(bart_machine)

#destroy BART model
destroy_bart_machine(bart_machine)

##Build another BART regression model
bart_machine = build_bart_machine(X,y, num_trees = 200, num_burn_in = 500,
num_iterations_after_burn_in =1000)

#Destroy BART model
destroy_bart_machine(bart_machine)

##Classification example

#get data and only use 2 factors
data(iris)
iris2 = iris[51:150,]
iris2$Species = factor(iris2$Species)

#build BART classification model
bart_machine = build_bart_machine(iris2[ ,1:4], iris2$Species)

##get estimated probabilities
phat = bart_machine$p_hat_train
##look at in-sample confusion matrix
bart_machine$confusion_matrix

#destroy BART model 
destroy_bart_machine(bart_machine)





base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("build_bart_machine", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("build_bart_machine_cv")
### * build_bart_machine_cv

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: build_bart_machine_cv
### Title: Build BART-CV
### Aliases: build_bart_machine_cv
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine_cv = build_bart_machine_cv(X, y)

#information about cross-validated model
summary(bart_machine_cv)

#destroy BART model
destroy_bart_machine(bart_machine_cv)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("build_bart_machine_cv", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("calc_credible_intervals")
### * calc_credible_intervals

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: calc_credible_intervals
### Title: Calculate Credible Intervals
### Aliases: calc_credible_intervals
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

#get credible interval
cred_int = calc_credible_intervals(bart_machine, X)
print(head(cred_int))

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("calc_credible_intervals", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("calc_prediction_intervals")
### * calc_prediction_intervals

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: calc_prediction_intervals
### Title: Calculate Prediction Intervals
### Aliases: calc_prediction_intervals
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

#get prediction interval
pred_int = calc_prediction_intervals(bart_machine, X)
print(head(pred_int))

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("calc_prediction_intervals", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("check_bart_error_assumptions")
### * check_bart_error_assumptions

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: check_bart_error_assumptions
### Title: Check BART Error Assumptions
### Aliases: check_bart_error_assumptions
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 300 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

#check error diagnostics
check_bart_error_assumptions(bart_machine)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("check_bart_error_assumptions", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("cov_importance_test")
### * cov_importance_test

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: cov_importance_test
### Title: Importance Test for Covariate(s) of Interest
### Aliases: cov_importance_test
### Keywords: ~kwd1 ~kwd2

### ** Examples

##regression example

##generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

##now test if X[, 1] affects Y nonparametrically under the BART model assumptions
cov_importance_test(bart_machine, covariates = c(1))
## note the plot and the printed p-value

##destroy BART model
destroy_bart_machine(bart_machine)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("cov_importance_test", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("destroy_bart_machine")
### * destroy_bart_machine

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: destroy_bart_machine
### Title: Destroy BART Model in Java
### Aliases: destroy_bart_machine
### Keywords: ~kwd1 ~kwd2

### ** Examples

##Generate Friedman Data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model and destroy it 
bart_machine = build_bart_machine(X, y)

##should be called when object is no longer needed 
##and before potentially removing the object from R
destroy_bart_machine(bart_machine) 




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("destroy_bart_machine", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("dummify_data")
### * dummify_data

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: dummify_data
### Title: Dummify Design Matrix
### Aliases: dummify_data
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate data
set.seed(11)
x1 = rnorm(20)
x2 = as.factor(ifelse(x1 > 0, "A", "B"))
x3 = runif(20)
X = data.frame(x1,x2,x3)
#dummify data
X_dummified = dummify_data(X)
print(X_dummified)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("dummify_data", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("get_sigsqs")
### * get_sigsqs

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: get_sigsqs
### Title: Get Posterior Error Variance Estimates
### Aliases: get_sigsqs
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 300 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

#get posterior sigma^2's after burn-in and plot
sigsqs = get_sigsqs(bart_machine, plot_hist = TRUE)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("get_sigsqs", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("get_var_counts_over_chain")
### * get_var_counts_over_chain

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: get_var_counts_over_chain
### Title: Get the Variable Inclusion Counts
### Aliases: get_var_counts_over_chain
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 10
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y, num_trees = 20)

#get variable inclusion counts
var_counts = get_var_counts_over_chain(bart_machine)
print(var_counts)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("get_var_counts_over_chain", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("get_var_props_over_chain")
### * get_var_props_over_chain

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: get_var_props_over_chain
### Title: Get the Variable Inclusion Proportions
### Aliases: get_var_props_over_chain
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 10
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y, num_trees = 20)

#Get variable inclusion proportions
var_props = get_var_props_over_chain(bart_machine)
print(var_props)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("get_var_props_over_chain", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("init_java_for_bart_machine_with_mem_in_mb")
### * init_java_for_bart_machine_with_mem_in_mb

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: init_java_for_bart_machine_with_mem_in_mb
### Title: Initialize a JVM with a pre-specified heap size
### Aliases: init_java_for_bart_machine_with_mem_in_mb
### Keywords: ~kwd1 ~kwd2

### ** Examples

##initialize a Java Virtual Machine with heap size of 3000MB
##this should be run before any BART models are built 
##init_java_for_bart_machine_with_mem_in_mb(3000) ##not run



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("init_java_for_bart_machine_with_mem_in_mb", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("interaction_investigator")
### * interaction_investigator

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: interaction_investigator
### Title: Explore Pairwise Interactions in BART Model
### Aliases: interaction_investigator
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 10
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y, num_trees = 20)

#investigate interactions
interaction_investigator(bart_machine)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("interaction_investigator", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("investigate_var_importance")
### * investigate_var_importance

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: investigate_var_importance
### Title: Explore Variable Inclusion Proportions in BART Model
### Aliases: investigate_var_importance
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 10
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y, num_trees = 20)

#investigate variable inclusion proportions
investigate_var_importance(bart_machine)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("investigate_var_importance", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("k_fold_cv")
### * k_fold_cv

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: k_fold_cv
### Title: Estimate Out-of-sample Error with K-fold Cross validation
### Aliases: k_fold_cv
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

#evaluate default BART on 5 folds
k_fold_val = k_fold_cv(X, y)
print(k_fold_val$rmse)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("k_fold_cv", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("pd_plot")
### * pd_plot

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: pd_plot
### Title: Partial Dependence Plot
### Aliases: pd_plot
### Keywords: ~kwd1 ~kwd2

### ** Examples


#Regression example

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

#partial dependence plot for quadratic term
pd_plot(bart_machine, "X3")

#destroy BART model
destroy_bart_machine(bart_machine)


#Classification example

#get data and only use 2 factors
data(iris)
iris2 = iris[51:150,]
iris2$Species = factor(iris2$Species)

#build BART classification model
bart_machine = build_bart_machine(iris2[ ,1:4], iris2$Species)

#partial dependence plot 
pd_plot(bart_machine, "Petal.Width")

#destroy BART model
destroy_bart_machine(bart_machine)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("pd_plot", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("plot_convergence_diagnostics")
### * plot_convergence_diagnostics

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: plot_convergence_diagnostics
### Title: Plot Convergence Diagnostics
### Aliases: plot_convergence_diagnostics
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

#plot convergence diagnostics
plot_convergence_diagnostics(bart_machine)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("plot_convergence_diagnostics", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("plot_y_vs_yhat")
### * plot_y_vs_yhat

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: plot_y_vs_yhat
### Title: Plot the fitted Versus Actual Response
### Aliases: plot_y_vs_yhat
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate linear data
set.seed(11)
n  = 500 
p = 3
X = data.frame(matrix(runif(n * p), ncol = p))
y = 3*X[ ,1] + 2*X[ ,2] +X[ ,3] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

##generate plot
plot_y_vs_yhat(bart_machine)

#generate plot with prediction bands
plot_y_vs_yhat(bart_machine, prediction_intervals = TRUE)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("plot_y_vs_yhat", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("predict")
### * predict

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: predict.bartMachine
### Title: Make a prediction on data using a BART object
### Aliases: predict.bartMachine
### Keywords: ~kwd1 ~kwd2

### ** Examples

#Regression example

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

##make predictions on the training data
y_hat = predict(bart_machine, X)

##destroy BART model
destroy_bart_machine(bart_machine)

#Classification example
data(iris)
iris2 = iris[51 : 150, ] #do not include the third type of flower for this example
iris2$Species = factor(iris2$Species)  
bart_machine = build_bart_machine(iris2[ ,1:4], iris2$Species)

##make probability predictions on the training data
p_hat = predict(bart_machine, X)

##make class predictions on test data
y_hat_class = predict(bart_machine, X, type = "class")

##make class predictions on test data conservatively for ''versicolor''
y_hat_class_conservative = predict(bart_machine, X, type = "class", prob_rule_class = 0.9)

##destroy BART model
destroy_bart_machine(bart_machine)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("predict", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("print")
### * print

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: print.bartMachine
### Title: Summarizes information about a 'bartMachine' object.
### Aliases: print.bartMachine
### Keywords: ~kwd1 ~kwd2

### ** Examples

#Regression example

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

##print out details
print(bart_machine)

##Also, the default print works too
bart_machine

##destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("print", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("rmse_by_num_trees")
### * rmse_by_num_trees

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: rmse_by_num_trees
### Title: Assess the Out-of-sample RMSE by Number of Trees
### Aliases: rmse_by_num_trees
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 200 
p = 10
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y, num_trees = 20)

#explore RMSE by number of trees
rmse_by_num_trees(bart_machine)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("rmse_by_num_trees", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("set_bart_machine_num_cores")
### * set_bart_machine_num_cores

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: set_bart_machine_num_cores
### Title: Set the Number of Cores for BART
### Aliases: set_bart_machine_num_cores
### Keywords: ~kwd1 ~kwd2

### ** Examples

## set all parallelized functions to use 4 cores
## set_bart_machine_num_cores(4) ##not run



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("set_bart_machine_num_cores", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("summary")
### * summary

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: summary.bartMachine
### Title: Summarizes information about a 'bartMachine' object.
### Aliases: summary.bartMachine
### Keywords: ~kwd1 ~kwd2

### ** Examples

#Regression example

#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

##print out details
summary(bart_machine)

##Also, the default print works too
bart_machine

##destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("summary", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("var_selection_by_permute_response_cv")
### * var_selection_by_permute_response_cv

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: var_selection_by_permute_response_cv
### Title: Perform Variable Selection Using Cross-validation Procedure
### Aliases: var_selection_by_permute_response_cv
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 150 
p = 100 ##95 useless predictors 
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model (not actually used in variable selection)
bart_machine = build_bart_machine(X, y)

#variable selection via cross-validation
var_sel_cv = var_selection_by_permute_response_cv(bart_machine, k_folds = 3)
print(var_sel_cv$best_method)
print(var_sel_cv$important_vars_cv)

#destroy BART model
destroy_bart_machine(bart_machine)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("var_selection_by_permute_response_cv", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("var_selection_by_permute_response_three_methods")
### * var_selection_by_permute_response_three_methods

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: var_selection_by_permute_response_three_methods
### Title: Perform Variable Selection using Three Threshold-based
###   Procedures
### Aliases: var_selection_by_permute_response_three_methods
### Keywords: ~kwd1 ~kwd2

### ** Examples

#generate Friedman data
set.seed(11)
n  = 300 
p = 20 ##15 useless predictors 
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model (not actuall used in variable selection)
bart_machine = build_bart_machine(X, y)

#variable selection
var_sel = var_selection_by_permute_response_three_methods(bart_machine)
print(var_sel$important_vars_local_names)
print(var_sel$important_vars_global_max_names)

#destroy BART model
destroy_bart_machine(bart_machine)
  


base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("var_selection_by_permute_response_three_methods", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
### * <FOOTER>
###
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
