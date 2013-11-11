\name{plot_y_vs_yhat}
\alias{plot_y_vs_yhat}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot the fitted Versus Actual Response
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
plot_y_vs_yhat(bart_machine, X = NULL, y = NULL, credible_intervals = FALSE, prediction_intervals = FALSE, interval_confidence_level = 0.95)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
}
  \item{X}{
%%     ~~Describe \code{X} here~~
}
  \item{y}{
%%     ~~Describe \code{y} here~~
}
  \item{credible_intervals}{
%%     ~~Describe \code{credible_intervals} here~~
}
  \item{prediction_intervals}{
%%     ~~Describe \code{prediction_intervals} here~~
}
  \item{interval_confidence_level}{
%%     ~~Describe \code{interval_confidence_level} here~~
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
function (bart_machine, X = NULL, y = NULL, credible_intervals = FALSE, 
    prediction_intervals = FALSE, interval_confidence_level = 0.95) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    if (credible_intervals && prediction_intervals) {
        stop("You cannot plot both credibility intervals and prediction intervals simultaneously.")
    }
    if (bart_machine$pred_type == "classification") {
        stop("You cannot plot y vs y_hat for classification.")
    }
    if (is.null(X) & is.null(y)) {
        X = bart_machine$X
        y = bart_machine$y
        y_hat = bart_machine$y_hat_train
        in_sample = TRUE
    }
    else {
        predict_obj = bart_predict_for_test_data(bart_machine, 
            X, y)
        y_hat = predict_obj$y_hat
        in_sample = FALSE
    }
    if (credible_intervals) {
        credible_intervals = calc_credible_intervals(bart_machine, 
            X, interval_confidence_level)
        ci_a = credible_intervals[, 1]
        ci_b = credible_intervals[, 2]
        y_in_ppi = y >= ci_a & y <= ci_b
        prop_ys_in_ppi = sum(y_in_ppi)/length(y_in_ppi)
        plot(y, y_hat, main = paste(ifelse(in_sample, "In-Sample", 
            "Out-of-Sample"), " Fitted vs. Actual Values\nwith ", 
            round(interval_confidence_level * 100), "\% Cred. Int.'s (", 
            round(prop_ys_in_ppi * 100, 2), "\% coverage)", sep = ""), 
            xlab = paste("Actual Values", sep = ""), ylab = "Fitted Values", 
            xlim = c(min(min(y), min(y_hat)), max(max(y), max(y_hat))), 
            ylim = c(min(min(y), min(y_hat)), max(max(y), max(y_hat))), 
            cex = 0)
        for (i in 1:bart_machine$n) {
            segments(y[i], ci_a[i], y[i], ci_b[i], col = "grey", 
                lwd = 0.1)
        }
        for (i in 1:bart_machine$n) {
            points(y[i], y_hat[i], col = ifelse(y_in_ppi[i], 
                "darkgreen", "red"), cex = 0.6, pch = 16)
        }
    }
    else if (prediction_intervals) {
        credible_intervals = calc_prediction_intervals(bart_machine, 
            X, interval_confidence_level)
        ci_a = credible_intervals[, 1]
        ci_b = credible_intervals[, 2]
        y_in_ppi = y >= ci_a & y <= ci_b
        prop_ys_in_ppi = sum(y_in_ppi)/length(y_in_ppi)
        plot(y, y_hat, main = paste(ifelse(in_sample, "In-Sample", 
            "Out-of-Sample"), " Fitted vs. Actual Values\nwith ", 
            round(interval_confidence_level * 100), "\% Pred. Int.'s (", 
            round(prop_ys_in_ppi * 100, 2), "\% coverage)", sep = ""), 
            xlab = paste("Actual Values", sep = ""), ylab = "Fitted Values", 
            xlim = c(min(min(y), min(y_hat)), max(max(y), max(y_hat))), 
            ylim = c(min(min(y), min(y_hat)), max(max(y), max(y_hat))), 
            cex = 0)
        for (i in 1:bart_machine$n) {
            segments(y[i], ci_a[i], y[i], ci_b[i], col = "grey", 
                lwd = 0.1)
        }
        for (i in 1:bart_machine$n) {
            points(y[i], y_hat[i], col = ifelse(y_in_ppi[i], 
                "darkgreen", "red"), cex = 0.6, pch = 16)
        }
    }
    else {
        plot(y, y_hat, main = "Fitted vs. Actual Values", xlab = "Actual Values", 
            ylab = "Fitted Values", col = "blue", xlim = c(min(min(y), 
                min(y_hat)), max(max(y), max(y_hat))), ylim = c(min(min(y), 
                min(y_hat)), max(max(y), max(y_hat))), )
    }
    abline(a = 0, b = 1, lty = 2)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
