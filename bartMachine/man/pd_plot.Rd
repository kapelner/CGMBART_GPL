\name{pd_plot}
\alias{pd_plot}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Partial Dependence Plot
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
pd_plot(bart_machine, j, levs = c(0.05, seq(from = 0.1, to = 0.9, by = 0.1), 0.95), lower_ci = 0.025, upper_ci = 0.975)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
}
  \item{j}{
%%     ~~Describe \code{j} here~~
}
  \item{levs}{
%%     ~~Describe \code{levs} here~~
}
  \item{lower_ci}{
%%     ~~Describe \code{lower_ci} here~~
}
  \item{upper_ci}{
%%     ~~Describe \code{upper_ci} here~~
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
function (bart_machine, j, levs = c(0.05, seq(from = 0.1, to = 0.9, 
    by = 0.1), 0.95), lower_ci = 0.025, upper_ci = 0.975) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    if (class(j) == "numeric" && (j < 1 || j > bart_machine$p)) {
        stop(paste("You must set j to a number between 1 and p =", 
            bart_machine$p))
    }
    else if (class(j) == "character" && !(j \%in\% bart_machine$training_data_features)) {
        stop("j must be the name of one of the training features (see \"<bart_model>$training_data_features\")")
    }
    else if (!(class(j) == "numeric" || class(j) == "character")) {
        stop("j must be a column number or column name")
    }
    x_j = bart_machine$model_matrix_training_data[, j]
    x_j_quants = quantile(x_j, levs)
    bart_predictions_by_quantile = array(NA, c(length(levs), 
        bart_machine$n, bart_machine$num_iterations_after_burn_in))
    for (q in 1:length(levs)) {
        x_j_quant = x_j_quants[q]
        test_data = bart_machine$X
        test_data[, j] = rep(x_j_quant, bart_machine$n)
        bart_predictions_by_quantile[q, , ] = bart_machine_get_posterior(bart_machine, 
            test_data)$y_hat_posterior_samples
        cat(".")
    }
    cat("\n")
    if (bart_machine$pred_type == "classification") {
        bart_predictions_by_quantile = qnorm(bart_predictions_by_quantile)
    }
    bart_avg_predictions_by_quantile_by_gibbs = array(NA, c(length(levs), 
        bart_machine$num_iterations_after_burn_in))
    for (q in 1:length(levs)) {
        for (g in 1:bart_machine$num_iterations_after_burn_in) {
            bart_avg_predictions_by_quantile_by_gibbs[q, g] = mean(bart_predictions_by_quantile[q, 
                , g])
        }
    }
    bart_avg_predictions_by_quantile = apply(bart_avg_predictions_by_quantile_by_gibbs, 
        1, mean)
    bart_avg_predictions_lower = apply(bart_avg_predictions_by_quantile_by_gibbs, 
        1, quantile, probs = lower_ci)
    bart_avg_predictions_upper = apply(bart_avg_predictions_by_quantile_by_gibbs, 
        1, quantile, probs = upper_ci)
    var_name = ifelse(class(j) == "character", j, bart_machine$training_data_features[j])
    ylab_name = ifelse(bart_machine$pred_type == "classification", 
        "Partial Effect (Probits)", "Partial Effect")
    plot(x_j_quants, bart_avg_predictions_by_quantile, type = "o", 
        main = "Partial Dependence Plot", ylim = c(min(bart_avg_predictions_lower, 
            bart_avg_predictions_upper), max(bart_avg_predictions_lower, 
            bart_avg_predictions_upper)), ylab = ylab_name, xlab = paste(var_name, 
            "plotted at specified quantiles"))
    lines(x_j_quants, bart_avg_predictions_lower, type = "o", 
        col = "blue")
    lines(x_j_quants, bart_avg_predictions_upper, type = "o", 
        col = "blue")
    invisible(list(x_j_quants = x_j_quants, bart_avg_predictions_by_quantile = bart_avg_predictions_by_quantile))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
