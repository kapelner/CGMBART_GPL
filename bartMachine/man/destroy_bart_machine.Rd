\name{destroy_bart_machine}
\alias{destroy_bart_machine}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Destroy BART Model
}
\description{
This function destroys a BART model by setting all Java pointers to null. (ADAM: CORRECT?)
}
\usage{
destroy_bart_machine(bart_machine)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
}
}
\details{
Removing a ``bart_machine'' object from \code{R} does not free heap space from Java. Since BART objects can consume a large amount of RAM, it is important to remove these objects by calling this function if they are no longer needed or many BART objects are being created.
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
function (bart_machine) 
{
    .jcall(bart_machine$java_bart_machine, "V", "destroy")
    bart_machine$bart_destroyed = TRUE
    .jcall("java/lang/System", "V", "gc")
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
