#These are the functions that simulate a given data model
#They must be named simu_data_mod__<NAME> and the name must be added to the 
#vector of data models below if you want to include it in the bake-off.
#You should use underscores for spaces for display purposes later

simulate_data_from_simulation_name = function(simulated_data_model_name){
	#get the function from the R environment by doing a string lookup
	simulated_data_model_function = get(paste("simu_data_mod__", simulated_data_model_name, sep = ""))
	#then just run the function... pretty simple
	simulated_data_model_function()
}

DEFAULT_N = 1000
DEFAULT_SIGMA = 1


simu_data_mod__just_noise_linear = function(N = DEFAULT_N, sigma = DEFAULT_SIGMA){
	X1 = rep(0, N) #just an intercept
	y = X1 + rnorm(N, 0, sigma) #ie linear model with $\beta_0 = 0, \beta_1 = 1$
	Xy = as.data.frame(cbind(X1, y))
	colnames(Xy) = c("x_1", "y")
	Xy	
}

simu_data_mod__univariate_linear = function(N = DEFAULT_N, sigma = DEFAULT_SIGMA){
	X1 = runif(N, 0, 100)
	y = X1 + rnorm(N, 0, sigma) #ie linear model with $\beta_0 = 0, \beta_1 = 1$
	Xy = as.data.frame(cbind(X1, y))
	colnames(Xy) = c("x_1", "y")
	Xy	
}

simu_data_mod__bivariate_linear = function(N = DEFAULT_N, sigma = DEFAULT_SIGMA){
	X1 = runif(N, 0, 100)
	X2 = runif(N, 0, 100)
	y = X1 + X2 + rnorm(N, 0, sigma) #ie linear model with $\beta_0 = 0, \beta_1 = 1$
	Xy = as.data.frame(cbind(X1, X2, y))
	colnames(Xy) = c("x_1", "x_2", "y")
	Xy	
}


simu_data_mod__simple_tree_structure_sigsq_hundredth = function(N = DEFAULT_N){
	simu_data_mod__simple_tree_structure(N, sqrt(0.01))
}

simu_data_mod__simple_tree_structure_sigsq_tenth = function(N = DEFAULT_N){
	simu_data_mod__simple_tree_structure(N, sqrt(0.1))
}

simu_data_mod__simple_tree_structure_sigsq_half = function(N = DEFAULT_N){
	simu_data_mod__simple_tree_structure(N, sqrt(0.5))
}

simu_data_mod__simple_tree_structure_sigsq_3 = function(N = DEFAULT_N){
	simu_data_mod__simple_tree_structure(N, sqrt(3))
}

simu_data_mod__simple_tree_structure_sigsq_5 = function(N = DEFAULT_N){
	simu_data_mod__simple_tree_structure(N, sqrt(5))
}

simu_data_mod__simple_tree_structure_sigsq_10 = function(N = DEFAULT_N){
	simu_data_mod__simple_tree_structure(N, sqrt(10))
}

simu_data_mod__simple_tree_structure_sigsq_30 = function(N = DEFAULT_N){
	simu_data_mod__simple_tree_structure(N, sqrt(30))
}

simu_data_mod__simple_tree_structure_sigsq_50 = function(N = DEFAULT_N){
	simu_data_mod__simple_tree_structure(N, sqrt(50))
}

simu_data_mod__simple_tree_structure_sigsq_100 = function(N = DEFAULT_N){
	simu_data_mod__simple_tree_structure(N, sqrt(100))
}

simu_data_mod__simple_tree_structure = function(N = DEFAULT_N, sigma = DEFAULT_SIGMA){
	#create a tree model
	X1 = runif(N, 0, 100)
	X2 = runif(N, 0, 100)
	X3 = runif(N, 0, 100)
	p = 3
	y = array(NA, N)
	
	for (i in 1 : N){
		y[i] = simple_tree_tag(c(X1[i], X2[i], X3[i]))
	}
	#now add the noise
	y = y + rnorm(N, 0, sigma)
	
	Xy = as.data.frame(cbind(X1, X2, X3, y))
	colnames(Xy) = c(paste("x_", 1 : p, sep = ""), "y")
	Xy
}

simple_tree_tag = function(x_vec){
	if (x_vec[1] < 30){
		if (x_vec[3] < 10){
			return(10)
		}
		else {
			return(30)
		}
	}
	else {
		if (x_vec[2] < 80){
			return(50)
		}
		else {
			return(70)
		}		
	}	
}

#create Friedman model
simu_data_mod__friedman = function(N = DEFAULT_N, p = 10, sigma = DEFAULT_SIGMA){	
	X = as.data.frame(matrix(runif(N * p, 0, 1), nrow = N, ncol = p))
	y = 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5] + rnorm(N, 0, sigma)
	Xy = cbind(X, y)
	colnames(Xy) = c(paste("x_", 1 : p, sep = ""), "y")
	Xy
}

###### to create data files just uncomment these and run
#write.csv(create_just_noise_data(), file = "datasets/r_just_noise.csv", row.names = FALSE)
#write.csv(create_linear_univariate_data(), file = "datasets/r_univariatelinear.csv", row.names = FALSE)
#write.csv(create_linear_bivariate_data(), file = "datasets/r_bivariatelinear.csv", row.names = FALSE)
write.csv(simu_data_mod__simple_tree_structure(), file = "datasets/r_treemodel.csv", row.names = FALSE)
#write.csv(create_friedman_data(), file = "datasets/r_friedman.csv", row.names = FALSE)