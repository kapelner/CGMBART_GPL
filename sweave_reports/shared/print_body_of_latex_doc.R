for (model_name in c(real_regression_data_sets, simulated_data_sets)){
	cat(paste("\n\n\\chapter{", gsub("_", " ", model_name), "}\n\n", sep = "")) 
	cat("\\pagebreak\n")
	
	for (num_trees in num_trees_of_interest){
		cat(paste("\n\n\\section{m = ", num_trees, "}\n\n", sep = "")) 
		for (num_burn_in in num_burn_ins_of_interest){
			
			for (num_iterations_after_burn_in in num_iterations_after_burn_ins_of_interest){
				cat(paste("\n\n\\subsection{N\\_B = ", num_burn_in, ", N\\_G = ", num_iterations_after_burn_in, "}\n\n", sep = ""))	
				for (alpha in alphas_of_interest){
					for (beta in betas_of_interest){		
						image_titles = c(
							"sigsqs_by_gibbs", 
							"sigsqs_hist",
							"tree_liks_by_gibbs",
							"tree_liks_hist",
							"tree_nodes",
							"tree_depths"
						)
						#now... this if for debugging only the tree model						
						image_titles = c(image_titles, "plot_mu_vals_t_1")
						image_titles = c(image_titles, "hist_mu_vals_t_1")
#						image_titles = c(image_titles, "mu_vals_t_1_b_4")
#						image_titles = c(image_titles, "mu_vals_t_1_b_5")
#						image_titles = c(image_titles, "mu_vals_t_1_b_6")
#						image_titles = c(image_titles, "mu_vals_t_1_b_7")
						#now the comparisons
						image_titles = c(image_titles, "yvyhat_bart", "yvyhat_R_BART")
#								"yvyhat_RF", 
#								"yvyhat_CART"
						#now produce the latex code for the images
						for (i in 1 : length(image_titles)){
							if (i %% 2 == 1){
								cat("\\begin{figure}\n")
								cat("\\centering\n")
							}
							cat("\\subfigure{\n") #[$\\alpha =$ ", as.character(alpha), " $\\beta =$ ", beta, "]
							cat(paste("\\includegraphics[width=3.0in,type=pdf,ext=.pdf,read=.pdf]{../output_plots/", model_name, "_", image_titles[i], "_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, "_alpha_", as.character(alpha), "_beta_", beta, "}\n", sep = ""))
							cat("}\n")
							if (i %% 2 == 0){
								cat("\\end{figure}\n")	
								cat("\\FloatBarrier\n\n")
							}
						}				
						
						#now we can load in the root splits
						#				csv_filename = paste("../", PLOTS_DIR, "/", model_name, "_first_splits_m_", num_trees, "_n_B_", num_burn_in, "_n_G_a_", num_iterations_after_burn_in, ".csv", sep = "")
						#				root_splits = read.csv(csv_filename, header = TRUE)
						#				cat(paste("\n\n\\paragraph{Root Splits}\n\n", sep = ""))
						#				cat("{\\tiny\\begin{verbatim}\n")
						#				for (i in 1 : nrow(root_splits)){
						#					cat(paste(root_splits[i, 1], ": ", root_splits[i, 2], "\n"))
						#				}
						#				cat("\\end{verbatim}}\n\n")				
					}
				}		
			}
		}
	}
}