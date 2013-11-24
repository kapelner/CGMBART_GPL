##bart vs bayestree speed runs

library(bartMachine)
library(BayesTree)
library(randomForest)

init_java_for_bart_machine_with_mem_in_mb(10000)

nlist = c(100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000)
p = 20

n = 20000
num_trees = 50

time_mat = matrix(nrow = length(nlist), ncol = 7)
colnames(time_mat) = c("Him", "Us1coremem", "Us1core", "Us4coremem", "Us4core", "Us4core_no_run", "rf")
rownames(time_mat) = as.character(nlist)

counter = 1

for (n in nlist){
	
	cat("n =", n, "\n")
	
	X = matrix(rnorm(n * p), nrow = n, ncol = p)
	beta = runif(p, -1, 1)
	y = as.numeric(X %*% beta + rnorm(n))
  
  t1 = Sys.time()
  rob_bart = bart(x.train = X, y.train = y, ntree = num_trees, nskip = 250, verbose = FALSE)
  t2 = Sys.time()
  rob_time = t2 - t1
  cat("Rob time =", rob_time, "\n")
  time_mat[as.character(n), "Him"] = rob_time
  
  set_bart_machine_num_cores(1) ## 1 core
  t3 = Sys.time()
  bart_machine = build_bart_machine(as.data.frame(X), y, num_trees = num_trees, mem_cache_for_speed = TRUE)
  t4 = Sys.time()
  destroy_bart_machine(bart_machine)
  our_time_1_m = t4 - t3
  cat("Our 1 core memcache time =", our_time_1_m, "\n")  
  time_mat[as.character(n), "Us1coremem"] = our_time_1_m
  
  set_bart_machine_num_cores(1) ## 1 core
  t5 = Sys.time()
  bart_machine = build_bart_machine(as.data.frame(X), y, num_trees = num_trees, mem_cache_for_speed = FALSE)
  t6 = Sys.time()
  destroy_bart_machine(bart_machine)
  our_time_1 = t6 - t5
  cat("Our 1 core time =", our_time_1, "\n") 
  time_mat[as.character(n), "Us1core"] = our_time_1
  
  set_bart_machine_num_cores(4) ## 4 core
  t7 = Sys.time()
  bart_machine = build_bart_machine(as.data.frame(X), y, num_trees = num_trees, mem_cache_for_speed = TRUE)
  t8 = Sys.time()
  destroy_bart_machine(bart_machine)
  our_time_4_m = t8 - t7
  cat("Our 4 core memcache time =", our_time_4_m, "\n")
  time_mat[as.character(n), "Us4coremem"] = our_time_4_m
  
  set_bart_machine_num_cores(4) ## 4 core
  t9 = Sys.time()
  bart_machine = build_bart_machine(as.data.frame(X), y, num_trees = num_trees, mem_cache_for_speed = FALSE)
  t10 = Sys.time()
  destroy_bart_machine(bart_machine)
  our_time_4 = t10 - t9
  cat("Our 4 core time =", our_time_4, "\n")
  time_mat[as.character(n), "Us4core"] = our_time_4
  
  set_bart_machine_num_cores(4) ## 4 core
  t11 = Sys.time()
  bart_machine = build_bart_machine(as.data.frame(X), y, num_trees = num_trees, mem_cache_for_speed = FALSE, run_in_sample = FALSE)
  t12 = Sys.time()
  destroy_bart_machine(bart_machine)
  our_time_4_no_run = t12 - t11
  cat("Our 4 core time with no run in sample =", our_time_4_no_run, "\n")
  time_mat[as.character(n), "Us4core_no_run"] = our_time_4_no_run  

  t13 = Sys.time()
  rf_mod = randomForest(as.data.frame(X), y)
  t14 = Sys.time()
  rf_time = t14 - t13
  cat("RF time =", rf_time, "\n")
  time_mat[as.character(n), "rf"] = rf_time  
  
  counter = counter + 1
}
# convert some minutes to seconds
#
#Him Us1coremem   Us1core Us4coremem   Us4core Us4core_no_run          rf
#100    0.8510489  0.8580489  0.997057  0.4680269  0.524029      0.2940171  0.09500599
#200    1.4550831  1.6200931  1.885108  0.8450489  1.021058      0.5480320  0.22801304
#500    3.8632209  3.4952002  3.748215  2.2011261  2.580148      1.3120751  0.81504703
#1000   9.3625350  6.9033949  8.162466  4.7252710  4.896280      2.2141271  2.05811787
#2000  27.1955550 15.0618620 15.588891  9.0035150  9.638551      4.5812621  5.77433014
#5000   1.2765230 34.1509531 40.133295 29.3106761 26.458513     13.3497629 24.74141502
#10000  3.0791761  1.3423934  1.419681 58.0483201  1.074195     28.4616270  1.36421136
#20000  8.2231537  3.3437746  3.519501  2.4779751  2.371069      1.1337149  4.85072743
#40000 19.3956594  8.7569175  8.355245  7.6074685  5.802599      2.9437350 17.97974505


time_mat[8 : 9, ] = time_mat[8 : 9, ] * 60

#Him  Us1coremem    Us1core  Us4coremem    Us4core Us4core_no_run           rf
#100      0.8510489   0.8580489   0.997057   0.4680269   0.524029      0.2940171 9.500599e-02
#200      1.4550831   1.6200931   1.885108   0.8450489   1.021058      0.5480320 2.280130e-01
#500      3.8632209   3.4952002   3.748215   2.2011261   2.580148      1.3120751 8.150470e-01
#1000     9.3625350   6.9033949   8.162466   4.7252710   4.896280      2.2141271 2.058118e+00
#2000    27.1955550  15.0618620  15.588891   9.0035150   9.638551      4.5812621 5.774330e+00
#5000     1.2765230  34.1509531  40.133295  29.3106761  26.458513     13.3497629 2.474142e+01
#10000    3.0791761   1.3423934   1.419681  58.0483201   1.074195     28.4616270 1.364211e+00
#20000  493.3892200 200.6264751 211.170078 148.6785040 142.264137     68.0228910 2.910436e+02
#40000 1163.7395620 525.4150519 501.314673 456.4481070 348.155913    176.6241019 1.078785e+03

time_mat[7, c(1,2,3,5,7)] = time_mat[7, c(1,2,3,5,7)] * 60
time_mat[6, 1] = time_mat[6, 1] * 60 

setwd("C:\\Users\\Kapelner\\Desktop\\Dropbox\\BART_package\\data")
save(time_mat, file = "time_runs.RData")

#c("Him", "Us1coremem", "Us1core", "Us4coremem", "Us4core", "Us4core_no_run", "rf")
#http://www.stat.columbia.edu/~tzheng/files/Rcolor.pdf
COLORS = c("red", "darkblue", "darkblue", "darkviolet", "darkviolet", "darkgreen", "darkorange")
LTYS = c(1, 1, 2, 1, 2, 1, 1) 

plot(nlist / 1000, time_mat[, 1] / 60, type = "o", col = COLORS[1], lty = LTYS[1], lwd = 3, xlab = "Sample Size (1000's)", ylab = "Minutes", ylim = c(0, 12))
for (j in 2 : 7){
	lines(nlist / 1000, time_mat[, j] / 60, type = "o", col = COLORS[j], lty = LTYS[j], lwd = 3)
}
legend(x = -2, y = 13, c("BayesTree", 
				"bartMachine (1 core,\n memcache)", 
				"bartMachine (1 core)", 
				"bartMachine (4 cores,\n memcache)", 
				"bartMachine (4 cores)", 
				"bartMachine (4 cores,\n no in-sample)", 
				"randomForest"
				), COLORS, lty = LTYS)

#zoomed in plot
windows()
plot(nlist, time_mat[, 1], type = "o", col = COLORS[1], lty = LTYS[1], lwd = 3, xlab = "Sample Size", ylab = "Seconds", ylim = c(0, 27), xlim = c(100, 2000))
for (j in 2 : 7){
	lines(nlist, time_mat[, j], type = "o", col = COLORS[j], lty = LTYS[j], lwd = 3)
}
legend(x = -2, y = 29, c("BayesTree", 
				"bartMachine (1 core,\n memcache)", 
				"bartMachine (1 core)", 
				"bartMachine (4 cores,\n memcache)", 
				"bartMachine (4 cores)", 
				"bartMachine (4 cores,\n no in-sample)", 
				"randomForest"
		), COLORS, lty = LTYS)
