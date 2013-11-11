\name{plot_convergence_diagnostics}
\alias{plot_convergence_diagnostics}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot Convergence Diagnostics
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
plot_convergence_diagnostics(bart_machine)
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
    par(mfrow = c(2, 2))
    if (bart_machine$pred_type == "regression") {
        plot_sigsqs_convergence_diagnostics(bart_machine)
    }
    plot_mh_acceptance_reject(bart_machine)
    plot_tree_num_nodes(bart_machine)
    plot_tree_depths(bart_machine)
    par(mfrow = c(1, 1))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
