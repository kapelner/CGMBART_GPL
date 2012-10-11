directory_where_code_is = "C:\\Users\\kapelner\\workspace\\CGMBART_GPL"
setwd(directory_where_code_is)

source("r_scripts/bart_package.R")
source("r_scripts/create_simulated_models.R")
graphics.off()

sigsqs = 10^seq(from = -3, to = 4, by = 0.5)

z_offs_by_sigsq = list()

for (sigsq in sigsqs){
	sigma = sqrt(sigsq)
	training_data = simu_data_mod__simple_tree_structure(sigma = sigma)	
	test_data = simu_data_mod__simple_tree_structure(sigma = sigma)	
	
	#load up a bart model
	bart_machine = build_a_bart_model(training_data, 
			num_trees = 1, 
			num_burn_in = 2000, 
			num_iterations_after_burn_in = 2000,
			alpha = 0.95, 
			beta = 2,
			print_tree_illustrations = FALSE,
			debug_log = TRUE)
	
	#get posterior samples
	post_samples_all = predict_and_calc_ppis(bart_machine, test_data)$y_hat_posterior_samples
	
	z_offs = array(NA, nrow(test_data))
	for (i in 1 : nrow(test_data)){
		y_real = simple_tree_tag(test_data[i, 1 : 4])
		post_samples = post_samples_all[i, ]
		post_avg = mean(post_samples)
		p_normal = shapiro.test(post_samples)$p.value
		z_off = (post_avg - y_real) / sd(post_samples)
		z_offs[i] = z_off
#		hist(post_samples, 
#				br = 100, 
##				xlim = c(min(y_real - 0.1, post_samples), max(y_real + 0.1, post_samples)), 
#				main = paste("i", i, "post avg", round(post_avg, 5), "y_real", y_real, "normal test pval", round(p_normal, 2), "z_off", round(z_off, 1)))
#		abline(v = y_real, col = "blue", lwd = 3)
#		abline(v = post_avg, col = "red")
#		readline()
	}
	
#	sort(z_offs)
#	hist(z_offs, br =100)
#	unique(z_offs)
	z_offs_by_sigsq[[as.character(sigsq)]] = unique(z_offs)
}

par(mfrow = c(5,3))
for (sigsq in sigsqs){
	hist(z_offs_by_sigsq[[as.character(sigsq)]], 
		main = paste("z scores for sigsq =", round(sigsq, 3)), 
		xlim = c(-4,4), 
		xlab = "",
		ylab = "",
		br = 100)
}
