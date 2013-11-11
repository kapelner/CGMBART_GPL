\name{init_java_for_bart_machine_with_mem_in_mb}
\alias{init_java_for_bart_machine_with_mem_in_mb}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Initialize a JVM with a specified heap size
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
init_java_for_bart_machine_with_mem_in_mb(bart_max_mem)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_max_mem}{
%%     ~~Describe \code{bart_max_mem} here~~
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
function (bart_max_mem) 
{
    jinit_params = paste("-Xmx", bart_max_mem, "m", sep = "")
    .jinit(parameters = jinit_params)
    for (dependency in JAR_DEPENDENCIES) {
        .jaddClassPath(paste(find.package("bartMachine"), "/java/", 
            dependency, sep = ""))
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
