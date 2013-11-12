\name{get_var_props_over_chain}
\alias{get_var_props_over_chain}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get the Variable Inclusion Proportions
}
\description{
Computes the variable inclusion proportions for a BART model. 
}
\usage{
get_var_props_over_chain(bart_machine, type = "splits")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{type}{
If ``splits'', then the proportion of times each variable is chosen for a splitting rule is computed. If ``trees'', then the proportion of times each variable appears in a tree is computed.
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
\code{\link{get_var_counts_over_chain}}
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (bart_machine, type = "splits") 
{
    attribute_props = .jcall(bart_machine$java_bart_machine, 
        "[D", "getAttributeProps", as.integer(BART_NUM_CORES), 
        type)
    names(attribute_props) = colnames(bart_machine$model_matrix_training_data)[1:bart_machine$p]
    attribute_props
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
