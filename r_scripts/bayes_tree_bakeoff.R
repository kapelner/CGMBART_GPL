library(BayesTree, lib.loc = "~/R/")
library(MASS)
data(Boston)

ntree = c(1,5,20,50,100,200)
nsim = 10
train=Boston[1:253,]
test=Boston[254:506,]

out = matrix(nrow = nsim * length(ntree), ncol = 2)
count = 1
for (i in 1:length(ntree)){
	for (j in 1:nsim){
		bart_mod = bart(
				x.train = train[,-14], 
				y.train = train[, 14], 
				x.test = test[,-14],
				ntree = ntree[i],
				nskip = 2000, 
				ndpost = 2000, 
				sigest = sd(train$medv)
		)
		yhat = bart_mod$yhat.test.mean
		y=test[, 14]
		sse=sum((y-yhat)^2)
		rmse=sqrt(sse/length(y))
		print(count)
		out[count, 1]=ntree[i]
		out[count, 2]=rmse
		print(count)
		count=count+1
	}	
}

tapply(out[,2],out[,1], mean)
write.csv(out, "GPCS.csv")


##GPCS
#1        5       20       50      100      200 
#6.469490 5.926735 5.980814 5.707251 5.709086 5.735302 

##GPC
#1        5        20       50       100      200 
#6.438752 6.136565 5.845272 5.685153 5.649807 5.700116 

##GP
#1        5        20       50       100      200 
#7.254750 6.511943 6.305006 5.705443 5.683827 5.701422 

