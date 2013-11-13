\name{investigate_var_importance}
\alias{investigate_var_importance}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Explore Variable Inclusion Proportions in BART Model
}
\description{
Explore the variable inclusion proportions for a BART model to learn about the relative influence of the different covariates. This function includes an option to generate a plot of the variable inclusion proportions.
}
\usage{
investigate_var_importance(bart_machine, type = "splits", plot = TRUE, num_replicates_for_avg = 5, num_trees_bottleneck = 20, num_var_plot = Inf, bottom_margin = 10)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{type}{
If ``splits'', then the proportion of times each variable is chosen for a splitting rule is computed. If ``trees'', then the proportion of times each variable appears in a tree is computed.
}
  \item{plot}{
If TRUE, a plot of the variable inclusion proportions is generated. 
}
  \item{num_replicates_for_avg}{
The number of replicates of BART to be used to generate variable inclusion proportions. Averaging across multiple BART models improves stability of the estimates. See Bleich et al. (2013) for more details. 
}
  \item{num_trees_bottleneck}{
Number of trees to be used in the sum-of-trees for computing the variable inclusion proportions. A small number of trees should be used to force the variables to compete for entry into the model. Chipman et al. (2010) recommend 20. See this reference for more details. 
}
  \item{num_var_plot}{
Number of variables to be shown on the plot. If ``Inf'', all variables are plotted.
}
  \item{bottom_margin}{
Parameter to adjust the bottom margin of the graph if labels are clipped. ADAM: How does it work? Which gives more space?
}
}
\details{

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
Chipman et al. (2010)
}
\author{
Adam Kapelner and Justin Bleich
}
\note{
In the plot, the red bars correspond to the standard error of the variable inclusion proportion estimates (since multiple replicates were used).
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{interaction_investigator}}
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (bart_machine, type = "splits", plot = TRUE, num_replicates_for_avg = 5, 
    num_trees_bottleneck = 20, num_var_plot = Inf, bottom_margin = 10) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    var_props = array(0, c(num_replicates_for_avg, bart_machine$p))
    for (i in 1:num_replicates_for_avg) {
        if (i == 1 & num_trees_bottleneck == bart_machine$num_trees) {
            var_props[i, ] = get_var_props_over_chain(bart_machine, 
                type)
        }
        else {
            bart_machine_dup = build_bart_machine(bart_machine$X, 
                bart_machine$y, num_trees = num_trees_bottleneck, 
                num_burn_in = bart_machine$num_burn_in, num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in, 
                cov_prior_vec = bart_machine$cov_prior_vec, run_in_sample = FALSE, 
                use_missing_data = bart_machine$use_missing_data, 
                use_missing_data_dummies_as_covars = bart_machine$use_missing_data_dummies_as_covars, 
                num_rand_samps_in_library = bart_machine$num_rand_samps_in_library, 
                replace_missing_data_with_x_j_bar = bart_machine$replace_missing_data_with_x_j_bar, 
                impute_missingness_with_rf_impute = bart_machine$impute_missingness_with_rf_impute, 
                impute_missingness_with_x_j_bar_for_lm = bart_machine$impute_missingness_with_x_j_bar_for_lm, 
                verbose = FALSE)
            var_props[i, ] = get_var_props_over_chain(bart_machine_dup, 
                type)
            destroy_bart_machine(bart_machine_dup)
        }
        cat(".")
    }
    cat("\n")
    avg_var_props = colMeans(var_props)
    names(avg_var_props) = bart_machine$training_data_features_with_missing_features
    sd_var_props = apply(var_props, 2, sd)
    names(sd_var_props) = bart_machine$training_data_features_with_missing_features
    if (num_var_plot == Inf) {
        num_var_plot = bart_machine$p
    }
    avg_var_props_sorted_indices = sort(avg_var_props, decreasing = TRUE, 
        index.return = TRUE)$ix
    avg_var_props = avg_var_props[avg_var_props_sorted_indices][1:num_var_plot]
    sd_var_props = sd_var_props[avg_var_props_sorted_indices][1:num_var_plot]
    if (plot) {
        par(mar = c(bottom_margin, 6, 3, 0))
        if (is.na(sd_var_props[1])) {
            moe = 0
        }
        else {
            moe = 1.96 * sd_var_props/sqrt(num_replicates_for_avg)
        }
        bars = barplot(avg_var_props, names.arg = names(avg_var_props), 
            las = 2, ylab = "Inclusion Proportion", col = "gray", 
            ylim = c(0, max(avg_var_props + moe)))
        conf_upper = avg_var_props + 1.96 * sd_var_props/sqrt(num_replicates_for_avg)
        conf_lower = avg_var_props - 1.96 * sd_var_props/sqrt(num_replicates_for_avg)
        segments(bars, avg_var_props, bars, conf_upper, col = rgb(0.59, 
            0.39, 0.39), lwd = 3)
        segments(bars, avg_var_props, bars, conf_lower, col = rgb(0.59, 
            0.39, 0.39), lwd = 3)
        par(mar = c(5.1, 4.1, 4.1, 2.1))
    }
    invisible(list(avg_var_props = avg_var_props, sd_var_props = sd_var_props))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
