print(xtable(simulation_results, caption = paste("All Simulation Results (n\\_sim = ", run_model_N_times, ")", sep = "")), 
		size = "\\tiny", 
		tabular.environment = "longtable", 
		floating = FALSE, 
		include.rownames = FALSE)
print(xtable(avg_simulation_results, caption = paste("Average Simulation Results (n\\_sim = ", run_model_N_times, ")", sep = "")), 
		size = "\\tiny", 
		tabular.environment = "longtable", 
		floating = FALSE, 
		include.rownames = FALSE)