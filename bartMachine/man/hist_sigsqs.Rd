\name{hist_sigsqs}
\alias{hist_sigsqs}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Create a Histogram of the Posterior Error Variance Estimates
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
hist_sigsqs(bart_machine, extra_text = NULL, data_title = "data_model", save_plot = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
}
  \item{extra_text}{
%%     ~~Describe \code{extra_text} here~~
}
  \item{data_title}{
%%     ~~Describe \code{data_title} here~~
}
  \item{save_plot}{
%%     ~~Describe \code{save_plot} here~~
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
function (bart_machine, extra_text = NULL, data_title = "data_model", 
    save_plot = FALSE) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    if (bart_machine$pred_type == "classification") {
        stop("There are no sigsq's for classification.")
    }
    sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
    num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
    num_burn_in = bart_machine$num_burn_in
    num_gibbs = bart_machine$num_gibbs
    num_trees = bart_machine$num_trees
    sigsqs_after_burnin = sigsqs[(length(sigsqs) - num_iterations_after_burn_in):length(sigsqs)]
    assign("sigsqs_after_burnin", sigsqs_after_burnin, .GlobalEnv)
    avg_sigsqs = mean(sigsqs_after_burnin, na.rm = TRUE)
    if (save_plot) {
        save_plot_function(bart_machine, "sigsqs_hist", data_title)
    }
    else {
        dev.new()
    }
    ppi_a = quantile(sigsqs_after_burnin, 0.025)
    ppi_b = quantile(sigsqs_after_burnin, 0.975)
    hist(sigsqs_after_burnin, br = 100, main = "Histogram of sigsqs after burn-in", 
        xlab = paste("sigsq  avg = ", round(avg_sigsqs, 1), ",  95\% PPI = [", 
            round(ppi_a, 1), ", ", round(ppi_b, 1), "]", sep = ""))
    abline(v = avg_sigsqs, col = "blue")
    abline(v = ppi_a, col = "yellow")
    abline(v = ppi_b, col = "yellow")
    if (save_plot) {
        dev.off()
    }
    invisible(sigsqs_after_burnin)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
