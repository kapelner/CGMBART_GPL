\name{plot_convergence_diagnostics}
\alias{plot_convergence_diagnostics}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot Convergence Diagnostics
}
\description{
A suite of plots to assess convergence diagonstics and features of the BART model. 
}
\usage{
plot_convergence_diagnostics(bart_machine,  plots = c("sigsqs", "mh_acceptance", "num_nodes", "tree_depths"))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
\item{plots}{
The list of plots to be displayed. The four options are: "sigsqs", "mh_acceptance", "num_nodes", "tree_depths".
}
}
\details{
The ``sigsqs'' option plots the posterior error variance estimates by the Gibbs sample number. This is a standard tool to assess convergence of MCMC algorithms. This option is not applicable to classification settings.

The ''mh_acceptance'' option plots the proportion of Metropolis-Hastings steps accepted ?? WHAT DOES THIS DO AGAIN 

The ``num_nodes'' option plots the average number of nodes across each tree in the sum-of-trees model by the Gibbs sample number.

The ``num_nodes'' option plots the average tree depth across each tree in the sum-of-trees model by the Gibbs sample number.\\newline


}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{

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


% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
