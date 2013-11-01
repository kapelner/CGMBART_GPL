##Generate Friedman data

gen_friedman_data = function(n, p, sigma){
  if(p < 5) stop("p must be greater than or equal to 5")
  X = matrix(runif(n * p ), nrow = n, ncol = p)
  y = 10 * sin(pi * X[ ,1] *X[,2]) + 20 *(X[,3] - .5)^2 + 10 * X[ ,4] + 5 * X[ ,5] + rnorm(n, 0, sigma)
  dat = data.frame(y,X)
  dat
}

fr_data = gen_friedman_data(500, 10, 1)
y = fr_data$y
X = fr_data[, 2: 11]

library(bartMachine)
set_bart_machine_num_cores(1)
init_java_for_bart_machine_with_mem_in_mb(2500)



interaction_investigator(bart_machine, num_replicates_for_avg = 20, num_var_plot = 20) 
investigate_var_importance(bart_machine, num_replicates_for_avg = 10)




##fr data 2
fr_data = gen_friedman_data(500, 100, 1)
y = fr_data$y
X = fr_data[, 2: 101]

test_dat = gen_friedman_data(500, 100, 1)

cov_prior = c(rep(5, times = 5), rep(1, times = 95)) 
bart_machine = build_bart_machine(X, y)
bart_machine_informed = build_bart_machine(X, y, cov_prior_vec = cov_prior)
bart_machine_informed

bart_predict_for_test_data(bart_machine, test_dat[, 2:101], test_dat$y)$rmse
bart_predict_for_test_data(bart_machine_informed, test_dat[, 2:101], test_dat$y)$rmse
