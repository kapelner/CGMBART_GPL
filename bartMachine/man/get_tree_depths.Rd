\name{get_tree_depths}
\alias{get_tree_depths}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get the Posterior Tree Depths
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
get_tree_depths(bart_machine)
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
    tree_depths_after_burn_in = NULL
    for (c in 1:bart_machine$num_cores) {
        tree_depths_after_burn_in_core = t(sapply(.jcall(bart_machine$java_bart_machine, 
            "[[I", "getDepthsForTreesInGibbsSampAfterBurnIn", 
            as.integer(c)), .jevalArray))
        tree_depths_after_burn_in = rbind(tree_depths_after_burn_in, 
            tree_depths_after_burn_in_core)
    }
    tree_depths_after_burn_in
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
