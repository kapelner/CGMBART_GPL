mhrec = read.csv("debug_output//mh_iterations_full_record.csv")
mhrec$tree_._likelihood = as.numeric(as.character(mhrec$tree_._likelihood))
plot(1 : 1000, mhrec[, "tree_._likelihood"])