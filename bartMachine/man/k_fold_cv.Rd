\name{k_fold_cv}
\alias{k_fold_cv}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Estimate Out-of-sample Error with K-fold Cross validation
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
k_fold_cv(X, y, k_folds = 5, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
%%     ~~Describe \code{X} here~~
}
  \item{y}{
%%     ~~Describe \code{y} here~~
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
function (X, y, k_folds = 5, ...) 
{
    y_levels = levels(y)
    if (class(y) == "numeric" || class(y) == "integer") {
        pred_type = "regression"
    }
    else if (class(y) == "factor" & length(y_levels) == 2) {
        pred_type = "classification"
    }
    n = nrow(X)
    Xpreprocess = pre_process_training_data(X)
    p = ncol(Xpreprocess)
    if (k_folds <= 1 || k_folds > n) {
        stop("The number of folds must be at least 2 and less than or equal to n, use \"Inf\" for leave one out")
    }
    if (k_folds == Inf) {
        k_folds = n
    }
    holdout_size = round(n/k_folds)
    split_points = seq(from = 1, to = n, by = holdout_size)[1:k_folds]
    if (pred_type == "regression") {
        L1_err = 0
        L2_err = 0
    }
    else {
        confusion_matrix = matrix(0, nrow = 3, ncol = 3)
        rownames(confusion_matrix) = c(paste("actual", y_levels), 
            "use errors")
        colnames(confusion_matrix) = c(paste("predicted", y_levels), 
            "model errors")
    }
    Xy = data.frame(Xpreprocess, y)
    for (k in 1:k_folds) {
        cat(".")
        holdout_index_i = split_points[k]
        holdout_index_f = ifelse(k == k_folds, n, split_points[k + 
            1] - 1)
        test_data_k = Xy[holdout_index_i:holdout_index_f, ]
        training_data_k = Xy[-c(holdout_index_i:holdout_index_f), 
            ]
        bart_machine_cv = build_bart_machine(training_data_k[, 
            1:p], training_data_k[, (p + 1)], run_in_sample = FALSE, 
            ...)
        predict_obj = bart_predict_for_test_data(bart_machine_cv, 
            test_data_k[, 1:p], test_data_k[, (p + 1)])
        destroy_bart_machine(bart_machine_cv)
        if (pred_type == "regression") {
            L1_err = L1_err + predict_obj$L1_err
            L2_err = L2_err + predict_obj$L2_err
        }
        else {
            confusion_matrix[1:2, 1:2] = confusion_matrix[1:2, 
                1:2] + table(test_data_k$y, predict_obj$y_hat)
        }
    }
    cat("\n")
    if (pred_type == "regression") {
        list(L1_err = L1_err, L2_err = L2_err, rmse = sqrt(L2_err/n), 
            PseudoRsq = 1 - L2_err/sum((y - mean(y))^2))
    }
    else {
        confusion_matrix[3, 1] = round(confusion_matrix[2, 1]/(confusion_matrix[1, 
            1] + confusion_matrix[2, 1]), 3)
        confusion_matrix[3, 2] = round(confusion_matrix[1, 2]/(confusion_matrix[1, 
            2] + confusion_matrix[2, 2]), 3)
        confusion_matrix[1, 3] = round(confusion_matrix[1, 2]/(confusion_matrix[1, 
            1] + confusion_matrix[1, 2]), 3)
        confusion_matrix[2, 3] = round(confusion_matrix[2, 1]/(confusion_matrix[2, 
            1] + confusion_matrix[2, 2]), 3)
        confusion_matrix[3, 3] = round((confusion_matrix[1, 2] + 
            confusion_matrix[2, 1])/sum(confusion_matrix[1:2, 
            1:2]), 3)
        list(confusion_matrix = confusion_matrix, misclassification_error = confusion_matrix[3, 
            3])
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
