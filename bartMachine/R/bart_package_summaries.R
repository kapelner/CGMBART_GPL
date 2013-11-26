##give summary info about bart
summary.bartMachine = function(bart_machine){
	if (is_bart_destroyed(bart_machine)){
		stop("This BART machine has been destroyed. Please recreate.")
	}	
	cat(paste("Bart Machine v", VERSION, ifelse(bart_machine$pred_type == "regression", " for regression", " for classification"), "\n\n", sep = ""))
	if (bart_machine$use_missing_data){
		cat("Missing data feature ON\n")
	}
	#first print out characteristics of the training data
	cat(paste("training data n =", bart_machine$n, "and p =", bart_machine$p, "\n"))
	
  ##build time
	ttb = as.numeric(bart_machine$time_to_build, units = "secs")
	if (ttb > 60){
		ttb = as.numeric(bart_machine$time_to_build, units = "mins")
		cat(paste("built in", round(ttb, 2), "mins on", bart_machine$num_cores, ifelse(bart_machine$num_cores == 1, "core,", "cores,"), bart_machine$num_trees, "trees,", bart_machine$num_burn_in, "burn in and", bart_machine$num_iterations_after_burn_in, "post. samples\n"))
	} else {
		cat(paste("built in", round(ttb, 1), "secs on", bart_machine$num_cores, ifelse(bart_machine$num_cores == 1, "core,", "cores,"), bart_machine$num_trees, "trees,", bart_machine$num_burn_in, "burn in and", bart_machine$num_iterations_after_burn_in, "post. samples\n"))
	}
	
	if (bart_machine$pred_type == "regression"){
		sigsq_est = sigsq_est(bart_machine) ##call private function
		cat(paste("\nsigsq est for y beforehand:", round(bart_machine$sig_sq_est, 3), "\n"))
		cat(paste("avg sigsq estimate after burn-in:", round(sigsq_est, 5), "\n"))
		
		if (bart_machine$run_in_sample){
			cat("\nin-sample statistics:\n")
			cat(paste(" L1 =", round(bart_machine$L1_err_train, 2), "\n",
							"L2 =", round(bart_machine$L2_err_train, 2), "\n",
							"rmse =", round(bart_machine$rmse_train, 2), "\n"),
					"Pseudo-Rsq =", round(bart_machine$PseudoRsq, 4))
			
			es = bart_machine$residuals
			normal_p_val = shapiro.test(es)$p.value
			cat("\np-val for shapiro-wilk test of normality of residuals:", round(normal_p_val, 5), "\n")
			
			centered_p_val = t.test(es)$p.value
			cat("p-val for zero-mean noise:", round(centered_p_val, 5), "\n")	
		} else {
			cat("\nno in-sample information available (use option run_in_sample = TRUE next time)\n")
		}		
	} else if (bart_machine$pred_type == "classification"){
		if (bart_machine$run_in_sample){
			cat("\nconfusion matrix:\n\n")
			print(bart_machine$confusion_matrix)
		} else {
			cat("\nno in-sample information available (use option run_in_sample = TRUE next time)\n")
		}		
	}
	cat("\n")
}

#alias for summary
print.bartMachine = function(x, ...){ #alias for summary
	summary(x)
}