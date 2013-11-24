##bart vs bayestree speed runs

library(bartMachine)
library(BayesTree)

init_java_for_bart_machine_with_mem_in_mb(10000)

nlist = c(100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000)
p = 20

n = 20000
ntree = 50

time_mat = matrix(nrow = length(nlist), ncol = 3)
colnames(time_mat) = c("Us", "Him", "Us_MC")

counter = 1
for(n in nlist){
  x = matrix(rnorm(n * p), nrow = n, ncol = p)
  beta = runif(p, -1, 1)
  y = as.numeric(x%*%beta + rnorm(n))
  
  
  set_bart_machine_num_cores(4) ## 1 core
  t0 = Sys.time()
  bart_machine = build_bart_machine(as.data.frame(x), y, num_trees = ntree, mem_cache_for_speed = T, 
                                    num_burn_in = 500, num_iterations_after_burn_in = 1000, verbose = T)
  t1 = Sys.time()
  rob_bart = bart(x.train = x, y.train = y, ntree = ntree, nskip = 500, ndpost = 1000, verbose = T)
  t2 = Sys.time()
  
  our_time = t1 - t0; print(our_time)
  rob_time = t2 - t1; print(rob_time)
  
  destroy_bart_machine(bart_machine)
  
  
# set_bart_machine_num_cores(4) ##4 cores
 # t0 = Sys.time()
  #bart_machine = build_bart_machine(as.data.frame(x), y, num_trees = 200, 
   #                                 num_burn_in = 500, num_iterations_after_burn_in = 1000, verbose = F)
  #t1 = Sys.time()
  #ur_time_mc = t1-t0; print(our_time_mc)
  
  #destroy_bart_machine(bart_machine)

  time_mat[counter, 1] = our_time
  time_mat[counter, 2] = rob_time
  #time_mat[counter, 3] = rob_time
  print(counter)
  counter = counter + 1 
  
}

time_mat[5,2] = time_mat[5,2]*60
time_mat[6:8,] = time_mat[6:8,]*60

save(time_mat, file = "time_runs.rdata")
load("C:/Users//jbleich/Dropbox/BART_package/time_runs.rdata")

plot(nlist, time_mat[,2], type = "o", col = "blue", lwd = 3, xlab = "Sample Size", ylab = "Seconds")
lines(nlist, time_mat[,1], type = "o", col = "red", lwd = 3)


