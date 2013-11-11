\name{plot_tree_depths}
\alias{plot_tree_depths}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot Tree Depths
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
plot_tree_depths(bart_machine)
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
    tree_depths_after_burn_in = get_tree_depths(bart_machine)
    num_after_burn_in_per_core = nrow(tree_depths_after_burn_in)
    plot(1:num_after_burn_in_per_core, rep(0, num_after_burn_in_per_core), 
        type = "n", main = "Tree Depth by Gibbs Sample After Burn-in", 
        xlab = "Gibbs Sample", ylab = paste("Tree Depth for all cores"), 
        ylim = c(0, max(tree_depths_after_burn_in)))
    for (t in 1:ncol(tree_depths_after_burn_in)) {
        lines(1:num_after_burn_in_per_core, tree_depths_after_burn_in[, 
            t], col = rgb(0.9, 0.9, 0.9))
    }
    lines(1:num_after_burn_in_per_core, apply(tree_depths_after_burn_in, 
        1, mean), col = "blue", lwd = 4)
    lines(1:num_after_burn_in_per_core, apply(tree_depths_after_burn_in, 
        1, min), col = "black")
    lines(1:num_after_burn_in_per_core, apply(tree_depths_after_burn_in, 
        1, max), col = "black")
    if (bart_machine$num_cores > 1) {
        for (c in 2:bart_machine$num_cores) {
            abline(v = (c - 1) * bart_machine$num_iterations_after_burn_in/bart_machine$num_cores, 
                col = "gray")
        }
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
