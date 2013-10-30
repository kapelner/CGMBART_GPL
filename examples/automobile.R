setwd("C:\\Users\\Kapelner\\workspace\\CGMBART_GPL")
setwd("C:\\Users\\jbleich\\workspace\\CGMBART_GPL")

library(bartMachine)

Xy = read.csv("datasets/r_automobile.csv")
Xy = Xy[!is.na(Xy$price), ] #kill rows without a response
Xy = na.omit(Xy) #kill any rows with missing data (we illustrate missing data features further in this file)
y = log(as.numeric(Xy$price))
Xy$price = log(Xy$price)


#now remove some variables and coerce some to numeric
Xy$make = NULL
Xy$num_doors = ifelse(X$num_doors == "two", 2, 4)
Xy$num_cylinders = ifelse(X$num_cylinders == "twelve", 12, ifelse(X$num_cylinders == "eight", 8, ifelse(X$num_cylinders == "six", 6, ifelse(X$num_cylinders == "five", 5, ifelse(X$num_cylinders == "four", 4, ifelse(X$num_cylinders == "three", 3, 2))))))

X = Xy
X$price = NULL

#head(X)
#dim(X)
#summary(X)
#hist(y, br = 30)
#hist(y, br = 30)

set_bart_machine_num_cores(1)
init_java_for_bart_machine_with_mem_in_mb(2500)

bart_machine = build_bart_machine(X, y)
bart_machine

plot_y_vs_yhat(bart_machine)


oos_stats = k_fold_cv(X, y, k_folds = 10)
oos_stats

#build another model with alpha beta to show depth of trees

vars_selected = var_selection_by_permute_response_three_methods(bart_machine, bottom_margin = 10, num_permute_samples=10) ##fix dot printing
names(vars_selected)
cv_vars = var_selection_by_permute_response_cv(bart_machine, k_folds = 2, num_reps_for_avg = 5, num_permute_samples = 20, num_trees_pred_cv = 50)



vars_selected$permute_mat[, 45] == 0
vars_selected$var_true_props_avg[45] == 0

pd_plot(bart_machine, j="horsepower")
#pd_plot(bart_machine, j="width")
#pd_plot(bart_machine, j="stroke")
#pd_plot(bart_machine, j="horsepower")


rmse_by_num_trees(bart_machine, num_replicates = 20)

bart_machine_cv = build_bart_machine_cv(X, y, verbose = T)
# BART CV win: k: 2 nu, q: 3, 0.9 m: 200
bart_machine_cv

#what is oosRMSE?
oos_stats = k_fold_cv(X, y, k_folds = 10, k = 2, nu = 3, q = 0.9, num_trees = 200)
oos_stats


check_bart_error_assumptions(bart_machine_cv)
plot_convergence_diagnostics(bart_machine_cv)

investigate_var_importance(bart_machine_cv, num_replicates_for_avg = 100)
ints = interaction_investigator(bart_machine_cv, num_replicates_for_avg = 200, bottom_margin = 20)



cov_importance_test(bart_machine_cv, covariates = c("body_style_wagon"))
cov_importance_test(bart_machine_cv, covariates = c("width"))
cov_importance_test(bart_machine_cv, covariates = c("body_style_wagon", "body_style_sedan", "fuel_system_1bbl", "fuel_system_spdi", "wheel_drive_4wd", "engine_type_dohc", "height", "engine_type_ohcf", "engine_type_ohcv", "compression_ratio", "fuel_system_mfi", "body_style_hatchback", "bore", "symboling", "stroke"))
cov_importance_test(bart_machine_cv)

#symboling,normalized_losses,make,fuel_type,aspiration,num_doors,body_style,wheel_drive,engine_location,wheel_base,length,width,height,curb_weight,engine_type,num_cylinders,engine_size,fuel_system,bore,stroke,compression_ratio,horsepower,peak_rpm,city_mpg,highway_mpg,price


pd_plot(bart_machine_cv, j = "horsepower")
pd_plot(bart_machine_cv, j = "width")
#pd_plot(bart_machine, j="stroke")
#pd_plot(bart_machine, j="horsepower")

#################################












#now let's play with missing data
setwd("C:\\Users\\Kapelner\\workspace\\CGMBART_GPL")

library(bartMachine)

Xy = read.csv("datasets/r_automobile.csv")
Xy = Xy[!is.na(Xy$price), ] #kill rows without a response
Xy = na.omit(Xy) #kill any rows with missing data (we illustrate missing data features further in this file)
y = log(as.numeric(Xy$price))
Xy$price = log(Xy$price)


#now remove some variables and coerce some to numeric
Xy$make = NULL
Xy$num_doors = ifelse(X$num_doors == "two", 2, 4)
Xy$num_cylinders = ifelse(X$num_cylinders == "twelve", 12, ifelse(X$num_cylinders == "eight", 8, ifelse(X$num_cylinders == "six", 6, ifelse(X$num_cylinders == "five", 5, ifelse(X$num_cylinders == "four", 4, ifelse(X$num_cylinders == "three", 3, 2))))))

X = Xy
X$price = NULL


set_bart_machine_num_cores(4)
init_java_for_bart_machine_with_mem_in_mb(5000)

bart_machine = build_bart_machine(X, y, verbose = T, debug_log = T, use_missing_data = TRUE, use_missing_data_dummies_as_covars = TRUE)
bart_machine
bart_machine$training_data_features_with_missing_features

#we ask the question: does missingness itself matter?
cov_importance_test(bart_machine, covariates = 47 : 92, num_permutations = 100)
#p-val = 0.47

#doesn't seem to matter so let's not use the extra missing covariates
bart_machine = build_bart_machine(X, y, verbose = T, debug_log = T, use_missing_data = TRUE)
bart_machine
bart_machine$training_data_features_with_missing_features

oos_stats = k_fold_cv(X, y, use_missing_data = TRUE, k_folds = 10)
oos_stats

investigate_var_importance(bart_machine, num_replicates_for_avg = 25)

#now let's do some variable selection
windows()
vars_selected = var_selection_by_permute_response_three_methods(bart_machine, bottom_margin = 10)
vars_selected$important_vars_pointwise_names
vars_selected$important_vars_simul_max_names
vars_selected$important_vars_simul_se_names

vars_selected_cv = var_selection_by_permute_response_cv(bart_machine)

#build model on just the vars selected by ptwise
X_dummified = dummify_data(X)

bart_machine_ptwise_vars = build_bart_machine(X_dummified[, vars_selected$important_vars_pointwise_names], y, verbose = T, debug_log = T, use_missing_data = TRUE)
bart_machine_ptwise_vars
bart_machine_ptwise_vars$training_data_features_with_missing_features

oos_stats_red_mod = k_fold_cv(X_dummified[, vars_selected$important_vars_pointwise_names], y, use_missing_data = TRUE, k_folds = 20)
oos_stats_red_mod

bart_machine_simul_max_vars = build_bart_machine(X_dummified[, vars_selected$important_vars_simul_max_names], y, verbose = T, debug_log = T, use_missing_data = TRUE)
bart_machine_simul_max_vars
bart_machine_simul_max_vars$training_data_features_with_missing_features

oos_stats_red_mod = k_fold_cv(X_dummified[, vars_selected$important_vars_simul_max_names], y, use_missing_data = TRUE, k_folds = 20)
oos_stats_red_mod

interactions = interaction_investigator(bart_machine_simul_max_vars)

#conclusion: did better than original model by cutting out crap

mod = lm(as.formula(paste("y ~ (", paste(vars_selected$important_vars_simul_max_names, collapse = "+"), ")^2")), data = X_dummified)
summary(mod)


mod = lm(y ~ ., data = X)
summary(mod)

##oosRMSE for OLS


k_folds = 10
Xy$fuel_system = NULL
n = nrow(Xy)
p = ncol(Xy) - 1

holdout_size = round(n / k_folds)
split_points = seq(from = 1, to = n, by = holdout_size)[1 : k_folds]

L1_err = 0
L2_err = 0

for (k in 1 : k_folds){
	cat(".")
	holdout_index_i = split_points[k]
	holdout_index_f = ifelse(k == k_folds, n, split_points[k + 1] - 1)
	
	test_data_k = Xy[holdout_index_i : holdout_index_f, ]
	training_data_k = Xy[-c(holdout_index_i : holdout_index_f), ]
	
 
	training_data_k$engine_location = NULL
	ols_mod = lm(price ~ ., training_data_k)
	
	y_hat = predict(ols_mod, test_data_k[, 1 : p])
	
	
	
	#tabulate errors
	L1_err = L1_err + sum(abs(y_hat - test_data_k[, (p + 1)]))
	L2_err = L2_err + sum((y_hat - test_data_k[, (p + 1)])^2)
}
cat("\n")

list(L1_err = L1_err, L2_err = L2_err, rmse = sqrt(L2_err / n), PseudoRsq = 1 - L2_err / sum((y - mean(y))^2))


#conclusion: can build a pretty damn good parametric model with first order interactions












































#1. Title: 1985 Auto Imports Database
#
#2. Source Information:
#		-- Creator/Donor: Jeffrey C. Schlimmer (Jeffrey.Schlimmer@a.gp.cs.cmu.edu)
#-- Date: 19 May 1987
#-- Sources:
#		1) 1985 Model Import Car and Truck Specifications, 1985 Ward's
#		Automotive Yearbook.
#		2) Personal Auto Manuals, Insurance Services Office, 160 Water
#		Street, New York, NY 10038 
#		3) Insurance Collision Report, Insurance Institute for Highway
#		Safety, Watergate 600, Washington, DC 20037
#		
#		3. Past Usage:
#		-- Kibler,~D., Aha,~D.~W., \& Albert,~M. (1989).  Instance-based prediction
#		of real-valued attributes.  {\it Computational Intelligence}, {\it 5},
#		51--57.
#		-- Predicted price of car using all numeric and Boolean attributes
#		-- Method: an instance-based learning (IBL) algorithm derived from a
#		localized k-nearest neighbor algorithm.  Compared with a
#		linear regression prediction...so all instances
#		with missing attribute values were discarded.  This resulted with
#		a training set of 159 instances, which was also used as a test
#		set (minus the actual instance during testing).
#		-- Results: Percent Average Deviation Error of Prediction from Actual
#		-- 11.84% for the IBL algorithm
#		-- 14.12% for the resulting linear regression equation
#		
#		4. Relevant Information:
#		-- Description
#		This data set consists of three types of entities: (a) the
#		specification of an auto in terms of various characteristics, (b)
#		its assigned insurance risk rating, (c) its normalized losses in use
#		as compared to other cars.  The second rating corresponds to the
#		degree to which the auto is more risky than its price indicates.
#		Cars are initially assigned a risk factor symbol associated with its
#		price.   Then, if it is more risky (or less), this symbol is
#		adjusted by moving it up (or down) the scale.  Actuarians call this
#		process "symboling".  A value of +3 indicates that the auto is
#		risky, -3 that it is probably pretty safe.
#		
#		The third factor is the relative average loss payment per insured
#		vehicle year.  This value is normalized for all autos within a
#		particular size classification (two-door small, station wagons,
#		sports/speciality, etc...), and represents the average loss per car
#		per year.
#		
#		-- Note: Several of the attributes in the database could be used as a
#		"class" attribute.
#		
#		5. Number of Instances: 205
#		
#		6. Number of Attributes: 26 total
#		-- 15 continuous
#		-- 1 integer
#		-- 10 nominal
#		
#		7. Attribute Information:     
#		Attribute:                Attribute Range:
#		------------------        -----------------------------------------------
#		1. symboling:                -3, -2, -1, 0, 1, 2, 3.
#		2. normalized-losses:        continuous from 65 to 256.
#		3. make:                     alfa-romero, audi, bmw, chevrolet, dodge, honda,
#		isuzu, jaguar, mazda, mercedes-benz, mercury,
#		mitsubishi, nissan, peugot, plymouth, porsche,
#		renault, saab, subaru, toyota, volkswagen, volvo
#		4. fuel-type:                diesel, gas.
#		5. aspiration:               std, turbo.
#		6. num-of-doors:             four, two.
#		7. body-style:               hardtop, wagon, sedan, hatchback, convertible.
#		8. drive-wheels:             4wd, fwd, rwd.
#		9. engine-location:          front, rear.
#		10. wheel-base:               continuous from 86.6 120.9.
#		11. length:                   continuous from 141.1 to 208.1.
#		12. width:                    continuous from 60.3 to 72.3.
#		13. height:                   continuous from 47.8 to 59.8.
#		14. curb-weight:              continuous from 1488 to 4066.
#		15. engine-type:              dohc, dohcv, l, ohc, ohcf, ohcv, rotor.
#		16. num-of-cylinders:         eight, five, four, six, three, twelve, two.
#		17. engine-size:              continuous from 61 to 326.
#		18. fuel-system:              1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.
#		19. bore:                     continuous from 2.54 to 3.94.
#		20. stroke:                   continuous from 2.07 to 4.17.
#		21. compression-ratio:        continuous from 7 to 23.
#		22. horsepower:               continuous from 48 to 288.
#		23. peak-rpm:                 continuous from 4150 to 6600.
#		24. city-mpg:                 continuous from 13 to 49.
#		25. highway-mpg:              continuous from 16 to 54.
#		26. price:                    continuous from 5118 to 45400.
#		
#		8. Missing Attribute Values: (denoted by "?")
#		Attribute #:   Number of instances missing a value:
#		2.             41
#		6.             2
#		19.            4
#		20.            4
#		22.            2
#		23.            2
#		26.            4
#		
# http://archive.ics.uci.edu/ml/datasets/Automobile