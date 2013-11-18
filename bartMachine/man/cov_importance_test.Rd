\name{cov_importance_test}
\alias{cov_importance_test}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Importance Test for Covariate(s) of Interest
}
\description{
This function tests \eqn{H_0}: These covariates of interest
do not affect the response under the assumptions of the BART 
model.
}
\usage{
cov_importance_test(bart_machine, covariates = NULL, 
num_permutations = 100, num_trees = NULL, plot = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
  An object of class ``bart_machine''.
}
  \item{covariates}{
	A vector indicating the covariates of interest you wish to test for affecting the response.
}
  \item{num_permutations}{
	The number of times to permute the covariates of interest and create a new BART model (see details).
}
  \item{num_trees}{
	The number of trees to use during the permutations of the covariates of interest (see details). This
	parameter is not thought to have an effect on the result if left in the reasonable range of 50-200. 
	Default is to use the number of trees in the \code{bart_machine} object.
}
  \item{plot}{
	If \code{TRUE}, this produces a histogram of the Pseudo-Rsq's / total misclassifcation error rates from
	the \code{num_permutations} BART models created with the \code{covariates} permuted. The plot also illustrates
	the observed Pseudo-Rsq's / total misclassifcation error rate from the original training data and indicates
	the test's p-value.
}
}
\details{
To test the importance of a covariate or a set of covariates of interest on the response, we generate 
\code{num_permutations} BART models with the covariate(s) of interest permuted (differently each time). 
On each run, we record a measure of fit. For regression we record Pseudo-Rsq; for classification we record
total misclassification error. A p-value can then be generated as follows. For regression, this is the number of 
permutation-sampled Pseudo-Rsq's greater than the observed Pseudo-Rsq divided by 
\code{num_permutations}. For classification, this is the number of permutation-sampled 
total misclassification errors less than the observed total misclassification error divided by \code{num_permutations}.
}
\value{
\item{permutation_samples_of_error}{A vector which records the error metric of the BART models with the covariates permuted (see details).}
\item{observed_error_estimate}{For regression, this is the Pseudo-Rsq on the original
training data set. For classification, this is the observed total misclassification error
on the original training data set.}
\item{pval}{The approximate p-value for this test (see details). 
}
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
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##regression example

##generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

##now test if X[, 1] affects Y non parametrically under the BART assumptions and model
cov_importance_test(bart_machine, covariates = c(1))
## note the plot and the printed p-value

destroy_bart_machine(bart_machine)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
