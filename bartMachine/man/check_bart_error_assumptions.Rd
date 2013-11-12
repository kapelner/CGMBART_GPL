\name{check_bart_error_assumptions}
\alias{check_bart_error_assumptions}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Check BART Error Assumptions
}
\description{
Diagnostic tools to assess whether the errors of the BART model for regression are normally distributed and homoskedastic, as assumed by the model. This function generates a normal quantile plot of the residuals with a Shapiro-Wilks p-value as well as a residual plot.
}
\usage{
check_bart_error_assumptions(bart_machine, alpha_normal_test = 0.05, alpha_hetero_test = 0.05, hetero_plot = "yhats")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{alpha_normal_test}{
Significance level for Shapiro-Wilks test for normality. 
}
  \item{alpha_hetero_test}{
KILL
}
  \item{hetero_plot}{
If ``yhats'', the residuals are plotted against the fitted values of the response. If ``ys'', the residuals are plotted against the actual values of the response.
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
%% ~put references to the literature/web site here ~
}
\author{
Adam Kapelner and Justin Bleich
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{plot_sigsqs_convergence_diagnostics}}, \code{\link{plot_convergence_diagnostics}}
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (bart_machine, alpha_normal_test = 0.05, alpha_hetero_test = 0.05, 
    hetero_plot = "yhats") 
{
    if (!(hetero_plot \%in\% c("ys", "yhats"))) {
        stop("You must specify the parameter \"hetero_plot\" as \"ys\" or \"yhats\"")
    }
    if (bart_machine$pred_type == "classification") {
        stop("There are no convergence diagnostics for classification.")
    }
    graphics.off()
    par(mfrow = c(2, 1))
    es = bart_machine$residuals
    y_hat = bart_machine$y_hat
    normal_p_val = shapiro.test(es)$p.value
    qqp(es, col = "blue", main = paste("Assessment of Normality\n", 
        "p-val for shapiro-wilk test of normality of residuals:", 
        round(normal_p_val, 3)), xlab = "Normal Q-Q plot for in-sample residuals\n(Theoretical Quantiles)")
    if (hetero_plot == "yhats") {
        plot(y_hat, es, main = paste("Assessment of Heteroskedasticity\nFitted vs residuals"), 
            xlab = "Fitted Values", ylab = "Residuals", col = "blue")
    }
    else if (hetero_plot == "ys") {
        plot(bart_machine$y, es, main = paste("Assessment of Heteroskedasticity\nFitted vs residuals"), 
            xlab = "Actual Values", ylab = "Residuals", col = "blue")
    }
    abline(h = 0, col = "black")
    par(mfrow = c(1, 1))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
