\name{init_java_for_bart_machine_with_mem_in_mb}
\alias{init_java_for_bart_machine_with_mem_in_mb}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Initialize a JVM with a pre-specified heap size
}
\description{
Initialize a Java virtual Machine (JVM) with a pre-specified heap size in MB.  
}
\usage{
init_java_for_bart_machine_with_mem_in_mb(bart_max_mem)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_max_mem}{
Java Virtual Machine heap size in MB
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
This function can only be called once per \code{R} session and must be called before \code{\link{build_bart_machine}}, else the default heap size will be used (TK: what is default?).
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
