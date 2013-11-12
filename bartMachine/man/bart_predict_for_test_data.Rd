\name{bart_predict_for_test_data}
\alias{bart_predict_for_test_data}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Predict for test data
}
\description{
Utility wrapper function for computing out-of-sample metrics for a BART model when the test set results are known.
}
\usage{
bart_predict_for_test_data(bart_machine, Xtest, ytest)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
An object of class ``bartMachine''.
}
  \item{Xtest}{
Design matrix for test data containing rows at which predictions are to be made.
}
  \item{ytest}{
Actual outcomes for test data.
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
Add predict
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (bart_machine, Xtest, ytest) 
{
    if (bart_machine$bart_destroyed) {
        stop("This BART machine has been destroyed. Please recreate.")
    }
    ytest_hat = predict(bart_machine, Xtest)
    if (bart_machine$pred_type == "regression") {
        n = nrow(Xtest)
        L2_err = sum((ytest - ytest_hat)^2)
        list(y_hat = ytest_hat, L1_err = sum(abs(ytest - ytest_hat)), 
            L2_err = L2_err, rmse = sqrt(L2_err/n), e = ytest - 
                ytest_hat)
    }
    else {
        confusion_matrix = as.data.frame(matrix(NA, nrow = 3, 
            ncol = 3))
        rownames(confusion_matrix) = c(paste("actual", bart_machine$y_levels), 
            "use errors")
        colnames(confusion_matrix) = c(paste("predicted", bart_machine$y_levels), 
            "model errors")
        confusion_matrix[1:2, 1:2] = as.integer(table(ytest, 
            ytest_hat))
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
        list(y_hat = ytest_hat, confusion_matrix = confusion_matrix)
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
