\name{build_bart_machine}
\alias{build_bart_machine}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Build a BART Model}
\description{
Builds a BART model for regression or classification
}
\usage{
build_bart_machine(X = NULL, y = NULL, Xy = NULL, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000, alpha = 0.95, beta = 2, k = 2, q = 0.9, nu = 3, prob_rule_class = 0.5, mh_prob_steps = c(2.5, 2.5, 4)/9, debug_log = FALSE, run_in_sample = TRUE, s_sq_y = "mse", cov_prior_vec = NULL, use_missing_data = FALSE, covariates_to_permute = NULL, num_rand_samps_in_library = 10000, use_missing_data_dummies_as_covars = FALSE, replace_missing_data_with_x_j_bar = FALSE, impute_missingness_with_rf_impute = FALSE, impute_missingness_with_x_j_bar_for_lm = TRUE, mem_cache_for_speed = TRUE, verbose = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
Dataframe or matrix of predictors. Factors are automatically converted to dummies interally. 
}
  \item{y}{
Vector of response variable. If \code{y} is \code{numeric} or \code{integer}, a BART model for regression is built. If \code{y} is a factor with two levels, a BART model for classification is built.
}
  \item{Xy}{
A dataframe or matrix of predictors and the response. The response column must be named ``y''. 
}
  \item{num_trees}{
The number of trees to be grown in the sum-of-trees model.
}
  \item{num_burn_in}{
Number of MCMC samples to be discarded as ``burn-in''.
}
  \item{num_iterations_after_burn_in}{
Number of MCMC samples to draw from the posterior distribution of \eqn{\hat{f}(x)}. 
}
  \item{alpha}{
Base hyperparameter in tree prior for whether a node is nonterminal or not.
}
  \item{beta}{
Power hyperparameter in tree prior for whether a node is nonterminal or not.
}
  \item{k}{
For regression, \code{k} determines the prior probability that \eqn{E(Y|X)} is contained in the interval \eqn{(y_{min}, y_{max})}, based on a normal distribution. For example, when \eqn{k=2}, the prior probability is 95\%. For classification, \code{k} determines the prior probability that \eqn{E(Y|X)} is between \eqn{(-3,3)}. Note that a larger value of \code{k} results in more shrinkage and a more conservative fit. 
}
  \item{q}{
Quantile of the prior on the error variance at which the data-based estimate is placed. Note that the larger the value of \code{q}, the more aggressive the fit as you are placing more prior weight on values lower than the data-based estimate. Not used for classification.
}
  \item{nu}{
Degrees of freedom for the inverse \eqn{\chi-squared} prior. Not used for classification.
}
  \item{prob_rule_class}{
Threshold for classification. Any observation with a conditional probability greater than \code{prob_class_rule} is assigned the ``positive'' outcome. Note that the first level of the response is treated as the ``negative'' outcome and the second is treated as the ``positive'' outcome.  
}
  \item{mh_prob_steps}{
Vector of prior probabilities for proposing changes to the tree structures: (GROW, PRUNE, CHANGE)
}
  \item{debug_log}{
If TRUE, additional information about the model construction are printed to a file in the working directory.
}
  \item{run_in_sample}{
If TRUE, in-sample statistics such as \eqn{\hat{f}(x)}, Pseudo-\eqn{R^2}, and RMSE are computed. Setting this to FALSE when not needed can decrease computation time. 
}
  \item{s_sq_y}{
If ``mse'', a data-based estimated of the error variance is computed as the MSE from ordinary least squares regression. If ``var''., the data-based estimate is computed as the variance of the response. Not used in classification. 
}
  \item{cov_prior_vec}{
Vector assigning relative weights to how often a particular variable should be proposed as a candidate for a split. The vector is internally normalized so that the weights sum to 1. Note that the length of this vector must equal the length of the design matrix after dummification and augmentation of indicators of missingness (if used). See Bleich et al. (2013) for more details on when to use this feature. 
}
  \item{use_missing_data}{
If TRUE, the missing data feature is used to automatically handle missing data without imputation. See Kapelner and Bleich (2013) for details. 
}
  \item{covariates_to_permute}{
??
}
  \item{num_rand_samps_in_library}{
Before building a BART model, samples from the Standard Normal and \eqn{\chi-squared(\nu)} are drawn to be used in the MCMC steps. This parameter determines the number of samples to be taken.  
}
  \item{use_missing_data_dummies_as_covars}{
If TRUE, additional indicator variables for whether or not an observation in a particular column is missing are included. See Kapelner and Bleich (2013) for details.
}
  \item{replace_missing_data_with_x_j_bar}{
  If TRUE,missing entries in \code{X} are imputed with average value or modal category.

}
  \item{impute_missingness_with_rf_impute}{
If TRUE, missing entries are filled in using the rf.impute() function from the \code{randomForest} library. 
}
  \item{impute_missingness_with_x_j_bar_for_lm}{
If TRUE, when computing the data-based estimate of \eqn{\sigma^2}, missing entries are imputed with average value or modal category.
}
  \item{mem_cache_for_speed}{
??
}
  \item{verbose}{
Prints information about progress of the algorithm to the screen. 
}
}
\details{
Returns an object of class ``bart_machine''. Note that this object persists in the Java heap until code{\link{destroy_bart_machine}} is called or the R session is terminated. 
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
Adam Kapelner and Justin Bleich
}
\note{
This function is parallelized and each core will create an independent MCMC chain of size \code{num_burn_in} \eqn{+} \code{num_iterations_after_burn_in}/bart_machine_num_cores.
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{link{destroy_bart_machine}}
\code{link{bart_machine_cv}}
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.


}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line