\name{get_sigsqs}
\alias{get_sigsqs}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get Posterior Error Variance Estimates 
}
\description{
Returns the posterior estimates of the error variance from the Gibbs samples.
}
\usage{
get_sigsqs(bart_machine, after_burn_in = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{after_burn_in}{
If TRUE, only samples from after the burn-in period are returned.
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
Adam Kapelner and Justin Bleich
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{hist_sigsqs}}
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (bart_machine, after_burn_in = TRUE) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    if (bart_machine$pred_type == "classification") {
        stop("There are no sigsq's for classification.")
    }
    sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
    if (after_burn_in) {
        num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
        num_burn_in = bart_machine$num_burn_in
        num_gibbs = bart_machine$num_gibbs
        num_trees = bart_machine$num_trees
        sigsqs[(length(sigsqs) - num_iterations_after_burn_in):length(sigsqs)]
    }
    else {
        sigsqs
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
