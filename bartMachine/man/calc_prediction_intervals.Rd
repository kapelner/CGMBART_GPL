\name{calc_prediction_intervals}
\alias{calc_prediction_intervals}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Calculate Prediction Intervals
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
calc_prediction_intervals(bart_machine, new_data, pi_conf = 0.95, normal_samples_per_gibbs_sample = 100)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
}
  \item{new_data}{
%%     ~~Describe \code{new_data} here~~
}
  \item{pi_conf}{
%%     ~~Describe \code{pi_conf} here~~
}
  \item{normal_samples_per_gibbs_sample}{
%%     ~~Describe \code{normal_samples_per_gibbs_sample} here~~
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
function (bart_machine, new_data, pi_conf = 0.95, normal_samples_per_gibbs_sample = 100) 
{
    new_data = pre_process_new_data(new_data, bart_machine)
    n_test = nrow(new_data)
    pi_lower_bd = array(NA, n_test)
    pi_upper_bd = array(NA, n_test)
    y_hat_posterior_samples = t(sapply(.jcall(bart_machine$java_bart_machine, 
        "[[D", "getGibbsSamplesForPrediction", .jarray(new_data, 
            dispatch = TRUE), as.integer(BART_NUM_CORES)), .jevalArray))
    sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")
    all_prediction_samples = array(NA, c(n_test, bart_machine$num_iterations_after_burn_in, 
        normal_samples_per_gibbs_sample))
    for (i in 1:n_test) {
        for (n_g in 1:bart_machine$num_iterations_after_burn_in) {
            y_hat_draw = y_hat_posterior_samples[i, n_g]
            sigsq_draw = sigsqs[n_g]
            all_prediction_samples[i, n_g, ] = rnorm(n = normal_samples_per_gibbs_sample, 
                mean = y_hat_draw, sd = sqrt(sigsq_draw))
        }
    }
    for (i in 1:n_test) {
        pi_lower_bd[i] = quantile(c(all_prediction_samples[i, 
            , ]), (1 - pi_conf)/2)
        pi_upper_bd[i] = quantile(c(all_prediction_samples[i, 
            , ]), (1 + pi_conf)/2)
    }
    cbind(pi_lower_bd, pi_upper_bd)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
