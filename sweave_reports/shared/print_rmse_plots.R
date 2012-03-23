for (num_burn_in in num_burn_ins_of_interest){
	for (num_iterations_after_burn_in in num_iterations_after_burn_ins_of_interest){
		for (num_trees in num_trees_of_interest){				
			for (data_set in c(real_regression_data_sets, simulated_data_sets)){
				plot_filename = paste(PLOTS_DIR, "/rmse_comp_", data_set, "_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, "_alpha_", alpha, "_beta_", beta, sep = "")
				cat("\\begin{figure}\n")
				cat("\\centering\n")
				cat(paste("\\includegraphics[width=5.0in,type=pdf,ext=.pdf,read=.pdf]{../", plot_filename, "}\n", sep = ""))
				cat("\\end{figure}\n")
				cat("\\FloatBarrier\n\n")							
			}
		}
	}
}