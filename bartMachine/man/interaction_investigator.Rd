\name{interaction_investigator}
\alias{interaction_investigator}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Explore Pairwise Interactions in BART Model
}
\description{
Explore the pairwise interaction counts for a BART model to learn about interactions fit by the model. This function includes an option to generate a plot of the pairwise interaction counts.
}
\usage{
interaction_investigator(bart_machine, plot = TRUE, num_replicates_for_avg = 5, num_trees_bottleneck = 20, num_var_plot = 50, cut_bottom = NULL, bottom_margin = 10)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{plot}{
If TRUE, a plot of the pairwise interaction counts is generated. 
}
  \item{num_replicates_for_avg}{
The number of replicates of BART to be used to generate variable inclusion proportions. Averaging across multiple BART models improves stability of the estimates.
}
  \item{num_trees_bottleneck}{
Number of trees to be used in the sum-of-trees model for computing pairwise interactions. A small number of trees should be used to force the variables to compete for entry into the model.
}
  \item{num_var_plot}{
Number of variables to be shown on the plot. If ``Inf'', all variables are plotted.
}
  \item{cut_bottom}{
ADAM ??
}
  \item{bottom_margin}{
Parameter to adjust the bottom margin of the graph if labels are clipped. ADAM: How does it work? Which gives more space?
}
}
\details{
An interaction between two variables is considered to occur whenever a path from the root node of a tree to any terminal node contains splits using those two variables.
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
Ed Reference? 
JSS Paper?
}
\author{
%%  ~~who you are~~
}
\note{
In the plot, the red bars correspond to the standard error of the variable inclusion proportion estimates (since multiple replicates were used).
This function is parallelized by the number of cores set in \code{\link{set_bart_machine_num_cores}}.
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{investigate_var_importance}}
}
\examples{
#generate Friedman data
set.seed(11)
n  = 200 
p = 10
X = data.frame(matrix(runif(n * p), ncol = p))
y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)

##build BART regression model
bart_machine = build_bart_machine(X, y, num_trees = 20)

#investigate interactions
interaction_investigator(bart_machine)

#destroy BART model
destroy_bart_machine(bart_machine)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
