\name{bart_machine_get_posterior}
\alias{bart_machine_get_posterior}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get Posterior
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
bart_machine_get_posterior(bart_machine, new_data)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
}
  \item{new_data}{
%%     ~~Describe \code{new_data} here~~
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
function (bart_machine, new_data) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    if (class(new_data) != "matrix" && class(new_data) != "data.frame") {
        stop("X needs to be a matrix or data frame with the same column names as the training data.")
    }
    if (!bart_machine$use_missing_data) {
        nrow_before = nrow(new_data)
        new_data = na.omit(new_data)
        if (nrow_before > nrow(new_data)) {
            cat(nrow_before - nrow(new_data), "rows omitted due to missing data\n")
        }
    }
    if (nrow(new_data) == 0) {
        stop("No rows to predict.\n")
    }
    java_bart_machine = bart_machine$java_bart_machine
    num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in
    n = nrow(new_data)
    new_data = pre_process_new_data(new_data, bart_machine)
    if (!bart_machine$use_missing_data) {
        M = matrix(0, nrow = nrow(new_data), ncol = ncol(new_data))
        for (i in 1:nrow(new_data)) {
            for (j in 1:ncol(new_data)) {
                if (is.missing(new_data[i, j])) {
                  M[i, j] = 1
                }
            }
        }
        if (sum(M) > 0) {
            cat("WARNING: missing data found in test data and BART was not built with missing data feature!\n")
        }
    }
    y_hat_posterior_samples = t(sapply(.jcall(bart_machine$java_bart_machine, 
        "[[D", "getGibbsSamplesForPrediction", .jarray(new_data, 
            dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))
    y_hat = rowMeans(y_hat_posterior_samples)
    list(y_hat = y_hat, X = new_data, y_hat_posterior_samples = y_hat_posterior_samples)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
