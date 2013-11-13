\name{calc_credible_intervals}
\alias{calc_credible_intervals}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Calculate Credible Intervals
}
\description{
Generates credible intervals for \eqn{\hat{f}(x)} for a specified set of observations.
}
\usage{
calc_credible_intervals(bart_machine, new_data, ci_conf = 0.95)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{new_data}{
A data frame containing observations at which credible intervals for \eqn{\hat{f}(x)} are to be computed.
}
  \item{ci_conf}{
Confidence level for credible intervals.
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
\code{\link{calc_prediction_intervals}}, \code{\link{bart_machine_get_posterior}}
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (bart_machine, new_data, ci_conf = 0.95) 
{
    new_data = pre_process_new_data(new_data, bart_machine)
    n_test = nrow(new_data)
    ci_lower_bd = array(NA, n_test)
    ci_upper_bd = array(NA, n_test)
    y_hat_posterior_samples = t(sapply(.jcall(bart_machine$java_bart_machine, 
        "[[D", "getGibbsSamplesForPrediction", .jarray(new_data, 
            dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))
    y_hat = rowMeans(y_hat_posterior_samples)
    for (i in 1:n_test) {
        ci_lower_bd[i] = quantile(sort(y_hat_posterior_samples[i, 
            ]), (1 - ci_conf)/2)
        ci_upper_bd[i] = quantile(sort(y_hat_posterior_samples[i, 
            ]), (1 + ci_conf)/2)
    }
    cbind(ci_lower_bd, ci_upper_bd)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
