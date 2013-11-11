\name{plot_mh_acceptance_reject}
\alias{plot_mh_acceptance_reject}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot the proportion of Metropolis-Hastings Acceptances
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
plot_mh_acceptance_reject(bart_machine)
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
    mh_acceptance_reject = get_mh_acceptance_reject(bart_machine)
    a_r_before_burn_in = mh_acceptance_reject[["a_r_before_burn_in"]]
    a_r_before_burn_in_avg_over_trees = rowSums(a_r_before_burn_in)/bart_machine$num_trees
    a_r_after_burn_in_avgs_over_trees = list()
    for (c in 1:bart_machine$num_cores) {
        a_r_after_burn_in = mh_acceptance_reject[["a_r_after_burn_in"]][[c]]
        a_r_after_burn_in_avgs_over_trees[[c]] = rowSums(a_r_after_burn_in)/bart_machine$num_trees
    }
    num_after_burn_in_per_core = length(a_r_after_burn_in_avgs_over_trees[[1]])
    num_gibbs_per_core = bart_machine$num_burn_in + num_after_burn_in_per_core
    plot(1:num_gibbs_per_core, rep(0, num_gibbs_per_core), ylim = c(0, 
        1), type = "n", main = "Percent Acceptance by Gibbs Sample", 
        xlab = "Gibbs Sample", ylab = "\% of Trees Accepting")
    abline(v = bart_machine$num_burn_in, col = "grey")
    points(1:bart_machine$num_burn_in, a_r_before_burn_in_avg_over_trees, 
        col = "grey")
    tryCatch(lines(loess.smooth(1:bart_machine$num_burn_in, a_r_before_burn_in_avg_over_trees), 
        col = "black", lwd = 4), error = function(e) {
        e
    })
    for (c in 1:bart_machine$num_cores) {
        points((bart_machine$num_burn_in + 1):num_gibbs_per_core, 
            a_r_after_burn_in_avgs_over_trees[[c]], col = COLORS[c])
        tryCatch(lines(loess.smooth((bart_machine$num_burn_in + 
            1):num_gibbs_per_core, a_r_after_burn_in_avgs_over_trees[[c]]), 
            col = COLORS[c], lwd = 4), error = function(e) {
            e
        })
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
