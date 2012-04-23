library(zipfR)
 
nu = 3
q = 0.9

#Since Y \in [-1, 1] => max variance Y = 1
upto = 0.08

zs = seq(from = 0.0001, to = upto, by = 0.0001)
lambdas = array(NA, length(zs))
for (i in 1 : length(zs)){
	z = zs[i]
	lambdas[i] = z * 2 / nu * Rgamma.inv(nu / 2, q)
}

plot(zs, lambdas, xlim = c(0, max(zs, lambdas)), ylim = c(0, max(zs, lambdas)), pch = ".")
abline(a = 0, b = 1, col = "blue")

#plot
slope = lambdas[length(zs)] / zs[length(zs)]
abline(a = 0, b = slope, col = "red")

scaled = read.csv("debug_output/y_and_y_trans.csv")
head(scaled)

par(mfrow = c(2,1))
hist(scaled[, 1], br = 100, main = "real y")
hist(scaled[, 2], br = 100, main = "transformed y")
summary(scaled[, 2])


range = 63.47850661044171
sse = 78.08348305045136

null_dist = rnorm(100000, 1000 / range^2, sqrt(2000) / range^2)
hist(null_dist, br = 1000)

hist(sigsqs_after_burnin, br = 100)
abline(v = 1, col = "blue", lwd = 3)

n = 1000
nu = 3
lambda = 4.834109966488992E-5
sse = 0.002913120622673349
alpha_sim = (nu + n) / 2
beta_sim = (nu * lambda + sse) / 2
range_sq = 63.47850661044171^2
		
library(pscl)
hist(rigamma(10000, alpha_sim, beta_sim), br = 100)
abline(v = 1, col = "blue", lwd = 2)


lambda = 4.834109966488992E-5
xs = rigamma(10000, 1.5, 1.5 * lambda) * 63.478^2
hist(xs, br = 100000, xlim = c(0, 5))

mean(xs)
sd(xs)


