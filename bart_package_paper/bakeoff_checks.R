### check bakeoff data.
setwd("C:\\Users\\Kapelner\\workspace\\CGMBART_GPL")

library(BayesTree)
library(bartMachine)
library(randomForest)
library(mlbench)

data(BostonHousing); boston = BostonHousing; boston = as.data.frame(cbind(boston[,14],boston[,1:13])); names(boston)[1] = "medv"
wine.white = read.csv("datasets/r_wine_white.csv", header = TRUE)
wine.red = read.csv("datasets/r_wine_red.csv", header = TRUE)     
ozone =  read.csv("datasets/r_ozone.csv", header = TRUE)
pole =  read.csv("datasets/r_pole.csv", header = TRUE)   #OK
triazine =  read.csv("datasets/r_triazine.csv", header = TRUE)
ankara = read.csv("datasets/r_ankara.csv", header = TRUE)            
baseball = read.csv("datasets/r_baseballsalary.csv", header = TRUE)  #works
compactiv =  read.csv("datasets/r_compactiv.csv", header = TRUE)

datalist_names = c("boston", "triazine", "ozone", "baseball", "wine.red", "ankara", "wine.white", "pole", "compactiv")

		
# put the datasets into a list
datalist = list()
for (i in 1:length(datalist_names)){
	 data = get(datalist_names[[i]])
	 for (j in 1 : ncol(data)){
		 data[, j] = as.numeric(data[, j])
	 }
	 if (nrow(data) > 1000){
		 data = data[1 : 1000, ]
	 }
	 datalist[[i]] = data
	 names(datalist)[i] = datalist_names[i]
	 print(datalist_names[i])
	 print(dim(datalist[[i]]))
}

NREP = 20
KFOLDS = 10

oos_rmse_results = array(NA, c(length(datalist_names), 3, NREP))
rownames(oos_rmse_results) = datalist_names
colnames(oos_rmse_results) = c("bartMachine", "BayesTree", "RF")

for (nrep in 1 : NREP){
	for (dname in datalist_names){
		data = datalist[[dname]]
		X = data[, 2 : ncol(data)]
		y = data[, 1]
		
		rmse_bart_machine = k_fold_cv(X, y, k_folds = KFOLDS, verbose = FALSE)
		oos_rmse_results[dname, 1, nrep] = rmse_bart_machine$rmse
		
		rmse_rbart = k_fold_cv_bayes_tree(X, y, k_folds = KFOLDS)
		oos_rmse_results[dname, 2, nrep] = rmse_rbart$rmse
		
		rmse_rf = k_fold_cv_rf(X, y, k_folds = KFOLDS)
		oos_rmse_results[dname, 3, nrep] = rmse_rf$rmse
		
	}	
	print(oos_rmse_results[,, nrep])	
}

apply(oos_rmse_results, c(1, 2), mean)

#         bartMachine   BayesTree          RF
#boston    4.83869922  5.03051573  4.51326319
#triazine  0.04497970  0.04898590  0.04560320
#ozone    49.95679351 49.09947950 51.17056335
#baseball  0.01700984  0.01650077  0.02367905


k_fold_cv_bayes_tree = function(X, y, k_folds = 5, ...){
	
	n = nrow(X)
	
	p = ncol(X)
	
	if (k_folds <= 1 || k_folds > n){
		stop("The number of folds must be at least 2 and less than or equal to n, use \"Inf\" for leave one out")
	}
	
	if (k_folds == Inf){ #leave-one-out
		k_folds = n
	}	
	
	holdout_size = round(n / k_folds)
	split_points = seq(from = 1, to = n, by = holdout_size)[1 : k_folds]
	
	L1_err = 0
	L2_err = 0
	
	Xy = data.frame(X, y) ##set up data
	
	for (k in 1 : k_folds){
		cat(".")
		holdout_index_i = split_points[k]
		holdout_index_f = ifelse(k == k_folds, n, split_points[k + 1] - 1)
		
		test_data_k = Xy[holdout_index_i : holdout_index_f, ]
		training_data_k = Xy[-c(holdout_index_i : holdout_index_f), ]
		
		
		#build bart object
		rbart = bart(x.train = training_data_k[, 1 : p], y.train = as.numeric(training_data_k[, (p + 1)]), x.test = test_data_k[, 1 : p], verbose = FALSE, ntree = 50)
		y_hat = rbart$yhat.test.mean
		
		#tabulate errors
		y_test_k = test_data_k[, (p + 1)]
		L1_err = L1_err + sum(abs(y_test_k - y_hat))
		L2_err = L2_err + sum((y_test_k - y_hat)^2)
	}
	cat("\n")
	list(L1_err = L1_err, L2_err = L2_err, rmse = sqrt(L2_err / n), PseudoRsq = 1 - L2_err / sum((y - mean(y))^2))	
}

k_fold_cv_rf = function(X, y, k_folds = 5, ...){
	
	n = nrow(X)
	
	p = ncol(X)
	
	if (k_folds <= 1 || k_folds > n){
		stop("The number of folds must be at least 2 and less than or equal to n, use \"Inf\" for leave one out")
	}
	
	if (k_folds == Inf){ #leave-one-out
		k_folds = n
	}	
	
	holdout_size = round(n / k_folds)
	split_points = seq(from = 1, to = n, by = holdout_size)[1 : k_folds]
	
	L1_err = 0
	L2_err = 0
	
	Xy = data.frame(X, y) ##set up data
	
	for (k in 1 : k_folds){
		cat(".")
		holdout_index_i = split_points[k]
		holdout_index_f = ifelse(k == k_folds, n, split_points[k + 1] - 1)
		
		test_data_k = Xy[holdout_index_i : holdout_index_f, ]
		training_data_k = Xy[-c(holdout_index_i : holdout_index_f), ]
		
		
		#build bart object
		rf = randomForest(x = training_data_k[, 1 : p], y = as.numeric(training_data_k[, (p + 1)]), verbose = FALSE)
		y_hat = predict(rf, test_data_k)
		
		#tabulate errors
		y_test_k = test_data_k[, (p + 1)]
		L1_err = L1_err + sum(abs(y_test_k - y_hat))
		L2_err = L2_err + sum((y_test_k - y_hat)^2)
	}
	cat("\n")
	list(L1_err = L1_err, L2_err = L2_err, rmse = sqrt(L2_err / n), PseudoRsq = 1 - L2_err / sum((y - mean(y))^2))	
}
