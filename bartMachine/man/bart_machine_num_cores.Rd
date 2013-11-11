\name{bart_machine_num_cores}
\alias{bart_machine_num_cores}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Return number of cores being used
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
bart_machine_num_cores()
}
%- maybe also 'usage' for other objects documented here.
\details{
Returns the number of cores currently being used by parallelized BART functions
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{BART_NUM_CORES}{number of cores currently being used by parallelized BART functions}
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
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function () 
{
    BART_NUM_CORES
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
