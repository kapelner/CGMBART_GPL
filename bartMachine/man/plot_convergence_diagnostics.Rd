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
plot_convergence_diagnostics(bart_machine,  
plots = c("sigsqs", "mh_acceptance", "num_nodes", "tree_depths"))
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

The ``num_nodes'' option plots the average tree depth across each tree in the sum-of-trees model by the Gibbs sample number.


}
\value{
None.
}
\references{

}
\author{
Adam Kapelner and Justin Bleich
}
\note{
ADAM: NOTES ABOUT multicore plots?
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
#generate Friedman data
set.seed(11)
n  = 200 
p = 5
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y)

#plot convergence diagnostics
plot_convergence_diagnostics(bart_machine)

#destroy BART model
destroy_bart_machine(bart_machine)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
