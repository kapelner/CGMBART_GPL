\name{rmse_by_num_trees}
\alias{rmse_by_num_trees}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Assess the Out-of-sample RMSE by Number of Trees
}
\description{
Assess out-of-sample RMSE of a BART model for varying numbers of trees in the sum-of-trees model.
}
\usage{
rmse_by_num_trees(bart_machine, tree_list = c(5, seq(10, 50, 10), 100, 150, 200), in_sample = FALSE, plot = TRUE, holdout_pctg = 0.3, num_replicates = 4)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{tree_list}{
List of sizes for the sum-of-trees models
}
  \item{in_sample}{
If TRUE, the RMSE is computed on in-sample data rather than an out-of-sample holdout. 
}
  \item{plot}{
If TRUE, a plot of the RMSE by the number of trees in the ensemble is created.
}
  \item{holdout_pctg}{
Percentage of the data to be treated as an out-of-sample holdout. 
}
  \item{num_replicates}{
Number of replicates to average the results over. Each replicate uses a randomly sampled holdout of the data, (which could have overlap).
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
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
%%  ~~who you are~~
}
\note{
Since using a large number of trees can substantially increase computation time, this plot can help assess whether a smaller ensemble size is sufficient to obtain desirable predictive performance.
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (bart_machine, tree_list = c(5, seq(10, 50, 10), 100, 
    150, 200), in_sample = FALSE, plot = TRUE, holdout_pctg = 0.3, 
    num_replicates = 4) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    if (bart_machine$pred_type == "classification") {
        stop("This function does not work for classification.")
    }
    X = bart_machine$X
    y = bart_machine$y
    n = bart_machine$n
    rmses = array(NA, c(num_replicates, length(tree_list)))
    cat("num_trees = ")
    for (t in 1:length(tree_list)) {
        for (r in 1:num_replicates) {
            if (in_sample) {
                bart_machine_dup = bart_machine_duplicate(bart_machine, 
                  num_trees = tree_list[t], run_in_sample = TRUE)
                rmses[r, t] = bart_machine_dup$rmse_train
            }
            else {
                holdout_indicies = sample(1:n, holdout_pctg * 
                  n)
                Xtrain = X[setdiff(1:n, holdout_indicies), ]
                ytrain = y[setdiff(1:n, holdout_indicies)]
                Xtest = X[holdout_indicies, ]
                ytest = y[holdout_indicies]
                bart_machine_dup = bart_machine_duplicate(bart_machine, 
                  Xtrain, ytrain, num_trees = tree_list[t])
                predict_obj = bart_predict_for_test_data(bart_machine_dup, 
                  Xtest, ytest)
                rmses[r, t] = predict_obj$rmse
            }
            destroy_bart_machine(bart_machine_dup)
            cat("..")
            cat(tree_list[t])
        }
    }
    cat("\n")
    rmse_means = colMeans(rmses)
    if (plot) {
        rmse_sds = apply(rmses, 2, sd)
        y_mins = rmse_means - 2 * rmse_sds
        y_maxs = rmse_means + 2 * rmse_sds
        plot(tree_list, rmse_means, type = "o", xlab = "Number of Trees", 
            ylab = paste(ifelse(in_sample, "In-Sample", "Out-Of-Sample"), 
                "RMSE"), main = paste(ifelse(in_sample, "In-Sample", 
                "Out-Of-Sample"), "RMSE by Number of Trees"), 
            ylim = c(min(y_mins), max(y_maxs)))
        if (num_replicates > 1) {
            for (t in 1:length(tree_list)) {
                lowers = rmse_means[t] - 1.96 * rmse_sds[t]/sqrt(num_replicates)
                uppers = rmse_means[t] + 1.96 * rmse_sds[t]/sqrt(num_replicates)
                segments(tree_list[t], lowers, tree_list[t], 
                  uppers, col = "grey", lwd = 0.1)
            }
        }
    }
    invisible(rmse_means)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
