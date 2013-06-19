test_heteroskedasticity_of_bart_model = function(bart_machine, num_permutations = 100, plot = TRUE, num_trees = 20, ...){
	test_heteroskedasticity_of_any_model(bart_machine$residuals, bart_machine$X, num_permutations, plot, num_trees, ...)
}

test_heteroskedasticity_of_any_model = function(es, X, num_permutations = 100, plot = TRUE, num_trees = 20, ...){
	#build a bart machine with the response being the log squared residuals
	bart_machine = build_bart_machine(X, log(es^2), run_in_sample = FALSE, ...)
	#then do a global test to see if any of the covariates matter when predicting the log squared residuals
	cov_importance_test(bart_machine, num_permutations = num_permutations, plot = plot, num_trees = num_trees)
}
