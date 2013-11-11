\name{build_bart_machine_cv}
\alias{build_bart_machine_cv}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Build BART-CV
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
build_bart_machine_cv(X = NULL, y = NULL, Xy = NULL, num_tree_cvs = c(50, 200), k_cvs = c(2, 3, 5), nu_q_cvs = list(c(3, 0.9), c(3, 0.99), c(10, 0.75)), k_folds = 5, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
%%     ~~Describe \code{X} here~~
}
  \item{y}{
%%     ~~Describe \code{y} here~~
}
  \item{Xy}{
%%     ~~Describe \code{Xy} here~~
}
  \item{num_tree_cvs}{
%%     ~~Describe \code{num_tree_cvs} here~~
}
  \item{k_cvs}{
%%     ~~Describe \code{k_cvs} here~~
}
  \item{nu_q_cvs}{
%%     ~~Describe \code{nu_q_cvs} here~~
}
  \item{k_folds}{
%%     ~~Describe \code{k_folds} here~~
}
  \item{\dots}{
%%     ~~Describe \code{\dots} here~~
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
function (X = NULL, y = NULL, Xy = NULL, num_tree_cvs = c(50, 
    200), k_cvs = c(2, 3, 5), nu_q_cvs = list(c(3, 0.9), c(3, 
    0.99), c(10, 0.75)), k_folds = 5, ...) 
{
    y_levels = levels(y)
    if (class(y) == "numeric" || class(y) == "integer") {
        pred_type = "regression"
    }
    else if (class(y) == "factor" & length(y_levels) == 2) {
        pred_type = "classification"
    }
    if (pred_type == "classification") {
        nu_q_cvs = list(c(3, 0.9))
    }
    if ((is.null(X) && is.null(Xy)) || is.null(y) && is.null(Xy)) {
        stop("You need to give BART a training set either by specifying X and y or by specifying a matrix Xy which contains the response named \"y.\"\n")
    }
    else if (is.null(X) && is.null(y)) {
        y = Xy$y
        Xy$y = NULL
        X = Xy
    }
    min_rmse_num_tree = NULL
    min_rmse_k = NULL
    min_rmse_nu_q = NULL
    min_oos_rmse = Inf
    min_oos_misclassification_error = Inf
    for (k in k_cvs) {
        for (nu_q in nu_q_cvs) {
            for (num_trees in num_tree_cvs) {
                cat(paste("  BART CV try: k:", k, "nu, q:", paste(as.numeric(nu_q), 
                  collapse = ", "), "m:", num_trees, "\n"))
                k_fold_results = k_fold_cv(X, y, k_folds = k_folds, 
                  num_trees = num_trees, k = k, nu = nu_q[1], 
                  q = nu_q[2], ...)
                if (pred_type == "regression" && k_fold_results$rmse < 
                  min_oos_rmse) {
                  min_oos_rmse = k_fold_results$rmse
                  min_rmse_k = k
                  min_rmse_nu_q = nu_q
                  min_rmse_num_tree = num_trees
                }
                else if (pred_type == "classification" && k_fold_results$misclassification_error < 
                  min_oos_misclassification_error) {
                  min_oos_misclassification_error = k_fold_results$misclassification_error
                  min_rmse_k = k
                  min_rmse_nu_q = nu_q
                  min_rmse_num_tree = num_trees
                }
            }
        }
    }
    cat(paste("  BART CV win: k:", min_rmse_k, "nu, q:", paste(as.numeric(min_rmse_nu_q), 
        collapse = ", "), "m:", min_rmse_num_tree, "\n"))
    build_bart_machine(X, y, num_trees = min_rmse_num_tree, k = min_rmse_k, 
        nu = min_rmse_nu_q[1], q = min_rmse_nu_q[2], ...)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
