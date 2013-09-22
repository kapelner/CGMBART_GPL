library(pscl)


nu = 3
lambda = 1.947915e-05
range_sq = 63.47^2
s_y_trans_sq = 0.0475254257
s_y_sq = s_y_trans_sq * range_sq


#what does the prior look like?
prior_dist = rigamma(500000, nu / 2, (nu * lambda) / 2)
median(prior_dist) * range_sq
mean(prior_dist)
hist(prior_dist, br = 100000, xlim = c(0, 0.1), xlab = "sigsq (censored at 0.2, true max ~ 200), blue indicates 90%ile")
abline(v = s_y_trans_sq, col = "blue", lwd = 2)
quantile(prior_dist, 0.9)
s_y_trans_sq

hist(prior_dist * range_sq, br = 100000, xlim = c(0,1000))
abline(v = s_y_sq, col = "blue", lwd = 2)
quantile(prior_dist * range_sq, 0.9)
s_y_sq

#okay the prior is right... what about the posterior??

n = 1000
sigsqs = 10^seq(from = -2, to = 3, by = 0.5)
ten_pctile_chisq_df_3 = qchisq(0.1, 3)

#make biases table
biases = matrix(NA, nrow = length(sigsqs), ncol = 2)
rownames(biases) = round(sigsqs, 2)
colnames(biases) = c("bias", "pct off")

graphics.off()
par(mfrow = c(3, 4))
N_sim = 650 #I don't want to psychological confuse this number with the more important one above, n

for (i in 1 : length(sigsqs)){
	sigsq = sigsqs[i]
	posterior_samps = array(NA, N_sim)
	for (n_sim in 1 : N_sim){
		#get ys and sample statistics of them by actually sampling the data again and again
		ys = simu_data_mod__simple_tree_structure(sigma = sqrt(sigsq))$y
		y_trans = (ys - min(ys)) / (max(ys) - min(ys)) - 9.5
		s_sq_y_trans = var(y_trans) / 10
		
		#calculate lambda here from quantile as per Abba's quick formula
		lambda = ten_pctile_chisq_df_3 / 3 * s_sq_y_trans
		range_sq = (max(ys) - min(ys))^2
		
		#sample an sse from the SSE distribution
		sse = sigsq / range_sq * rchisq(1, n)  #rnorm(1, n * sigsq / range_sq, sqrt(2 * n) * sigsq / range_sq)
		
		#calc shape and scale params
		beta = (sse + nu * lambda) / 2
		alpha = (nu + n) / 2
		posterior_samps[n_sim] = rigamma(1, alpha, beta) * range_sq
	}
	
	hist(posterior_samps, 
		xlim = c(min(posterior_samps, sigsq), max(posterior_samps, sigsq)), 
		main = paste("posterior sigsq with n = ", n, ", sigsq = ", round(sigsq, 2), "\nsimple tree model", sep = ""),
		br = 100)
	abline(v = sigsq, col = "blue", lwd = 2)
	
	bias = sigsq - mean(posterior_samps)
	biases[i, ] = c(bias, bias / sigsq * 100)
}
