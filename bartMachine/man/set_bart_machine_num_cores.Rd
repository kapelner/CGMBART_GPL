\name{set_bart_machine_num_cores}
\alias{set_bart_machine_num_cores}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
none
}
\description{
Sets the number of cores to be used for all parallelized BART functions.
}
\usage{
set_bart_machine_num_cores(num_cores)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{num_cores}{
Number of cores to use
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
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (num_cores) 
{
    assign("BART_NUM_CORES", num_cores, ".GlobalEnv")
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
