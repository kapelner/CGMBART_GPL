\name{calc_prediction_intervals}
\alias{calc_prediction_intervals}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Calculate Prediction Intervals
}
\description{
Generates prediction intervals for \eqn{\hat{y}} for a specified set of observations.
}
\usage{
calc_prediction_intervals(bart_machine, new_data, pi_conf = 0.95, 
normal_samples_per_gibbs_sample = 100)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{new_data}{
A data frame containing observations at which prediction intervals for \eqn{\hat{y}} are to be computed.
}
  \item{pi_conf}{
Confidence level for prediction intervals.
}
  \item{normal_samples_per_gibbs_sample}{
ADAM FILL IN
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
Returns a matrix of the lower and upper bounds of the prediction intervals for each observation in \code{new_data}.
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
Adam Kapelner and Justin Bleich
}
\note{
This function is parallelized by the number of cores set in \code{\link{set_bart_machine_num_cores}}.
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{calc_credible_intervals}}, \code{\link{bart_machine_get_posterior}}
}
\examples{
#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

#get prediction interval
pred_int = calc_prediction_intervals(bart_machine, X)
print(head(pred_int))

#destroy BART model
destroy_bart_machine(bart_machine)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
