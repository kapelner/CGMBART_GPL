\name{plot_sigsqs_convergence_diagnostics}
\alias{plot_sigsqs_convergence_diagnostics}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot Error Variance Estimates to Assess Convergence
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
plot_sigsqs_convergence_diagnostics(bart_machine)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
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
function (bart_machine) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    if (bart_machine$pred_type == "classification") {
        stop("There are no convergence diagnostics for classification.")
    }
    sigsqs = get_sigsqs(bart_machine, after_burn_in = FALSE)
    num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
    num_burn_in = bart_machine$num_burn_in
    num_gibbs = bart_machine$num_gibbs
    num_trees = bart_machine$num_trees
    sigsqs_after_burnin = sigsqs[(length(sigsqs) - num_iterations_after_burn_in):length(sigsqs)]
    avg_sigsqs_after_burn_in = mean(sigsqs_after_burnin, na.rm = TRUE)
    plot(sigsqs, main = paste("Sigsq Estimates over Gibbs Samples"), 
        xlab = "Gibbs sample (yellow lines: after burn-in 95\% PPI)", 
        ylab = paste("Sigsq by iteration, avg after burn-in =", 
            round(avg_sigsqs_after_burn_in, 3)), ylim = c(quantile(sigsqs, 
            0.01), quantile(sigsqs, 0.99)), pch = ".", cex = 3, 
        col = "gray")
    points(sigsqs, pch = ".", col = "red")
    ppi_sigsqs = quantile(sigsqs[num_burn_in:length(sigsqs)], 
        c(0.025, 0.975))
    abline(a = ppi_sigsqs[1], b = 0, col = "yellow")
    abline(a = ppi_sigsqs[2], b = 0, col = "yellow")
    abline(a = avg_sigsqs_after_burn_in, b = 0, col = "blue")
    abline(v = num_burn_in, col = "gray")
    if (bart_machine$num_cores > 1) {
        for (c in 2:bart_machine$num_cores) {
            abline(v = num_burn_in + (c - 1) * bart_machine$num_iterations_after_burn_in/bart_machine$num_cores, 
                col = "gray")
        }
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
