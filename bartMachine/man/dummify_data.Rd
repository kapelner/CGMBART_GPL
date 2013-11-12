\name{dummify_data}
\alias{dummify_data}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Dummify Design Matrix
}
\description{
Create a dataframe with factors converted to dummies. 
}
\usage{
dummify_data(data, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{
Data frame to be dummified. 
}
  \item{\dots}{
Additional parameters. For internal processing only. 
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
BART handles dummification internally. This function is provided as a utility function. 
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
function (data, ...) 
{
    as.data.frame(pre_process_training_data(data, ...))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
