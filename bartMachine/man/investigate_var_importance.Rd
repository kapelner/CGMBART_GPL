\name{investigate_var_importance}
\alias{investigate_var_importance}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Explore Variable Inclusion Proportions in BART Model
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
investigate_var_importance(bart_machine, type = "splits", plot = TRUE, num_replicates_for_avg = 5, num_trees_bottleneck = 20, num_var_plot = Inf, bottom_margin = 10)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
}
  \item{type}{
%%     ~~Describe \code{type} here~~
}
  \item{plot}{
%%     ~~Describe \code{plot} here~~
}
  \item{num_replicates_for_avg}{
%%     ~~Describe \code{num_replicates_for_avg} here~~
}
  \item{num_trees_bottleneck}{
%%     ~~Describe \code{num_trees_bottleneck} here~~
}
  \item{num_var_plot}{
%%     ~~Describe \code{num_var_plot} here~~
}
  \item{bottom_margin}{
%%     ~~Describe \code{bottom_margin} here~~
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
%%  ~~further notes~~
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
