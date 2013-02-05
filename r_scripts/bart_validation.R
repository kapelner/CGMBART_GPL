k_fold_cv = function(bart_machine, k_folds = 5){
	if (bart_machine$bart_destroyed){
		stop("This BART machine has been destroyed. Please recreate.")
	}
	
	if (k_folds = Inf){ #leave-one-out
		k_folds = bart_machine$n
	}
	
	training_data = bart_machine$training_data
	n = bart_machine$n
	
	partition = floor(n / k_folds)
	
	for (k in 1 : k_folds){
		split
	}
	
}