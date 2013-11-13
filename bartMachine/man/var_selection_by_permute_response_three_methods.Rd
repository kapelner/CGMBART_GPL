\name{var_selection_by_permute_response_three_methods}
\alias{var_selection_by_permute_response_three_methods}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Perform Variable Selection using Three Threshold-based Procedures
}
\description{
Performs variable selection using the three thresholding methods introduced in Bleich et al. (2013). 
}
\usage{
var_selection_by_permute_response_three_methods(bart_machine, num_reps_for_avg = 10, num_permute_samples = 100, num_trees_for_permute = 20, alpha = 0.05, plot = TRUE, num_var_plot = Inf, bottom_margin = 10)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{num_reps_for_avg}{
Number of replicates to over over to for the BART model's variable inclusion proportions.
}
  \item{num_permute_samples}{
Number of permutations of the response to be made to generate the ``null'' permutation distribution.
}
  \item{num_trees_for_permute}{
Number of trees to use in the variable selection procedure. As with \code{\link{investigate_var_importance}}, a small number of trees should be used to force variables to compete for entry into the model. Note that this number is used to estimate both the ``true'' and ``null'' variable inclusion proportions.
}
  \item{alpha}{
Cut-off level for the thresholds.
}
  \item{plot}{
If TRUE, a plot showing which variables are selected by each of the procedures is generated.
}
  \item{num_var_plot}{
Number of variables (in order of decreasing variable inclusion proportion) to be plotted.
}
  \item{bottom_margin}{
Parameter to adjust the bottom margin of the graph if labels are clipped. ADAM: How does it work? Which gives more space?

}
}
\details{
TO DO. 
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
Bleich et al. (2013)
}
\author{
Adam Kapelner and Justin Bleich
}
\note{
Although the reference only explores regression settings, this procedure is applicable to both regression and classification problems. 
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{var_selection_by_permute_response_cv}}, \code{\link{investigate_var_importance}}
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (bart_machine, num_reps_for_avg = 10, num_permute_samples = 100, 
    num_trees_for_permute = 20, alpha = 0.05, plot = TRUE, num_var_plot = Inf, 
    bottom_margin = 10) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    permute_mat = matrix(NA, nrow = num_permute_samples, ncol = bart_machine$p)
    colnames(permute_mat) = bart_machine$training_data_features_with_missing_features
    cat("avg")
    var_true_props_avg = get_averaged_true_var_props(bart_machine, 
        num_reps_for_avg, num_trees_for_permute)
    var_true_props_avg = sort(var_true_props_avg, decreasing = TRUE)
    cat("null")
    for (b in 1:num_permute_samples) {
        permute_mat[b, ] = get_null_permute_var_importances(bart_machine, 
            num_trees_for_permute)
    }
    cat("\n")
    permute_mat = permute_mat[, names(var_true_props_avg)]
    pointwise_cutoffs = apply(permute_mat, 2, quantile, probs = 1 - 
        alpha)
    important_vars_pointwise_names = names(var_true_props_avg[var_true_props_avg > 
        pointwise_cutoffs & var_true_props_avg > 0])
    important_vars_pointwise_col_nums = sapply(1:length(important_vars_pointwise_names), 
        function(x) {
            which(important_vars_pointwise_names[x] == bart_machine$training_data_features_with_missing_features)
        })
    max_cut = quantile(apply(permute_mat, 1, max), 1 - alpha)
    important_vars_simul_max_names = names(var_true_props_avg[var_true_props_avg >= 
        max_cut & var_true_props_avg > 0])
    important_vars_simul_max_col_nums = sapply(1:length(important_vars_simul_max_names), 
        function(x) {
            which(important_vars_simul_max_names[x] == bart_machine$training_data_features_with_missing_features)
        })
    perm_se = apply(permute_mat, 2, sd)
    perm_mean = apply(permute_mat, 2, mean)
    cover_constant = bisectK(tol = 0.01, coverage = 1 - alpha, 
        permute_mat = permute_mat, x_left = 1, x_right = 20, 
        countLimit = 100, perm_mean = perm_mean, perm_se = perm_se)
    important_vars_simul_se_names = names(var_true_props_avg[which(var_true_props_avg >= 
        perm_mean + cover_constant * perm_se & var_true_props_avg > 
        0)])
    important_vars_simul_se_col_nums = sapply(1:length(important_vars_simul_se_names), 
        function(x) {
            which(important_vars_simul_se_names[x] == bart_machine$training_data_features_with_missing_features)
        })
    if (plot) {
        par(mar = c(bottom_margin, 6, 3, 0))
        if (num_var_plot == Inf | num_var_plot > bart_machine$p) {
            num_var_plot = bart_machine$p
        }
        par(mfrow = c(2, 1))
        non_zero_idx = which(var_true_props_avg > 0)[1:min(num_var_plot, 
            length(which(var_true_props_avg > 0)))]
        plot_n = length(non_zero_idx)
        if (length(non_zero_idx) < length(var_true_props_avg)) 
            warning(paste(length(which(var_true_props_avg == 
                0)), "covariates with inclusion proportions of 0 omitted from plots."))
        plot(1:plot_n, var_true_props_avg[non_zero_idx], type = "n", 
            xlab = NA, xaxt = "n", ylim = c(0, max(max(var_true_props_avg), 
                max_cut * 1.1)), main = "Local Procedure", ylab = "proportion included")
        axis(1, at = 1:plot_n, labels = names(var_true_props_avg[non_zero_idx]), 
            las = 2)
        for (j in non_zero_idx) {
            points(j, var_true_props_avg[j], pch = ifelse(var_true_props_avg[j] <= 
                quantile(permute_mat[, j], 1 - alpha), 1, 16))
        }
        sapply(non_zero_idx, function(s) {
            segments(s, 0, x1 = s, quantile(permute_mat[, s], 
                1 - alpha), col = "forestgreen")
        })
        plot(1:plot_n, var_true_props_avg[non_zero_idx], type = "n", 
            xlab = NA, xaxt = "n", ylim = c(0, max(max(var_true_props_avg), 
                max_cut * 1.1)), main = "Simul. Max and SE Procedures", 
            ylab = "proportion included")
        axis(1, at = 1:plot_n, labels = names(var_true_props_avg[non_zero_idx]), 
            las = 2)
        abline(h = max_cut, col = "red")
        for (j in non_zero_idx) {
            points(j, var_true_props_avg[j], pch = ifelse(var_true_props_avg[j] < 
                max_cut, ifelse(var_true_props_avg[j] > perm_mean[j] + 
                cover_constant * perm_se[j], 8, 1), 16))
        }
        sapply(non_zero_idx, function(s) {
            segments(s, 0, x1 = s, perm_mean[s] + cover_constant * 
                perm_se[s], col = "blue")
        })
        par(mar = c(5.1, 4.1, 4.1, 2.1))
        par(mfrow = c(1, 1))
    }
    invisible(list(important_vars_local_names = important_vars_pointwise_names, 
        important_vars_global_max_names = important_vars_simul_max_names, 
        important_vars_global_se_names = important_vars_simul_se_names, 
        important_vars_local_col_nums = as.numeric(important_vars_pointwise_col_nums), 
        important_vars_global_max_col_nums = as.numeric(important_vars_simul_max_col_nums), 
        important_vars_global_se_col_nums = as.numeric(important_vars_simul_se_col_nums), 
        var_true_props_avg = var_true_props_avg, permute_mat = permute_mat))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
