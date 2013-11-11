\name{check_bart_error_assumptions}
\alias{check_bart_error_assumptions}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Check BART Error Assumptions
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
check_bart_error_assumptions(bart_machine, alpha_normal_test = 0.05, alpha_hetero_test = 0.05, hetero_plot = "yhats")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
}
  \item{alpha_normal_test}{
%%     ~~Describe \code{alpha_normal_test} here~~
}
  \item{alpha_hetero_test}{
%%     ~~Describe \code{alpha_hetero_test} here~~
}
  \item{hetero_plot}{
%%     ~~Describe \code{hetero_plot} here~~
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
