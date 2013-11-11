\name{var_selection_by_permute_response_cv}
\alias{var_selection_by_permute_response_cv}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Perform Variable Selection Using Cross-validation Method
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
var_selection_by_permute_response_cv(bart_machine, k_folds = 5, num_reps_for_avg = 5, num_permute_samples = 100, num_trees_for_permute = 20, alpha = 0.05, num_trees_pred_cv = 200)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{bart_machine}{
%%     ~~Describe \code{bart_machine} here~~
}
  \item{k_folds}{
%%     ~~Describe \code{k_folds} here~~
}
  \item{num_reps_for_avg}{
%%     ~~Describe \code{num_reps_for_avg} here~~
}
  \item{num_permute_samples}{
%%     ~~Describe \code{num_permute_samples} here~~
}
  \item{num_trees_for_permute}{
%%     ~~Describe \code{num_trees_for_permute} here~~
}
  \item{alpha}{
%%     ~~Describe \code{alpha} here~~
}
  \item{num_trees_pred_cv}{
%%     ~~Describe \code{num_trees_pred_cv} here~~
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
function (bart_machine, k_folds = 5, num_reps_for_avg = 5, num_permute_samples = 100, 
    num_trees_for_permute = 20, alpha = 0.05, num_trees_pred_cv = 200) 
{
    if (k_folds <= 1 || k_folds > bart_machine$n) {
        stop("The number of folds must be at least 2 and less than or equal to n, use \"Inf\" for leave one out")
    }
    if (k_folds == Inf) {
        k_folds = bart_machine$n
    }
    holdout_size = round(bart_machine$n/k_folds)
    split_points = seq(from = 1, to = bart_machine$n, by = holdout_size)[1:k_folds]
    L2_err_mat = matrix(NA, nrow = k_folds, ncol = 3)
    colnames(L2_err_mat) = c("important_vars_local_names", "important_vars_global_max_names", 
        "important_vars_global _se_names")
    for (k in 1:k_folds) {
        cat("cv #", k, "\n", sep = "")
        holdout_index_i = split_points[k]
        holdout_index_f = ifelse(k == k_folds, bart_machine$n, 
            split_points[k + 1] - 1)
        training_X_k = bart_machine$model_matrix_training_data[-c(holdout_index_i:holdout_index_f), 
            ]
        training_y_k = y[-c(holdout_index_i:holdout_index_f)]
        bart_machine_temp = build_bart_machine(as.data.frame(training_X_k), 
            training_y_k, num_trees = bart_machine$num_trees, 
            num_burn_in = bart_machine$num_burn_in, cov_prior_vec = bart_machine$cov_prior_vec, 
            run_in_sample = FALSE, use_missing_data = bart_machine$use_missing_data, 
            use_missing_data_dummies_as_covars = bart_machine$use_missing_data_dummies_as_covars, 
            num_rand_samps_in_library = bart_machine$num_rand_samps_in_library, 
            replace_missing_data_with_x_j_bar = bart_machine$replace_missing_data_with_x_j_bar, 
            impute_missingness_with_rf_impute = bart_machine$impute_missingness_with_rf_impute, 
            impute_missingness_with_x_j_bar_for_lm = bart_machine$impute_missingness_with_x_j_bar_for_lm, 
            verbose = FALSE)
        bart_variables_select_obj_k = var_selection_by_permute_response_three_methods(bart_machine_temp, 
            num_permute_samples = num_permute_samples, num_trees_for_permute = num_trees_for_permute, 
            alpha = alpha, plot = FALSE)
        destroy_bart_machine(bart_machine_temp)
        test_X_k = bart_machine$model_matrix_training_data[holdout_index_i:holdout_index_f, 
            ]
        text_y_k = y[holdout_index_i:holdout_index_f]
        cat("method")
        for (method in colnames(L2_err_mat)) {
            cat(".")
            vars_selected_by_method = bart_variables_select_obj_k[[method]]
            if (length(vars_selected_by_method) == 0) {
                ybar_est = mean(training_y_k)
                L2_err_mat[k, method] = sum((text_y_k - ybar_est)^2)
            }
            else {
                training_X_k_red_by_vars_picked_by_method = as.data.frame(training_X_k[, 
                  vars_selected_by_method])
                bart_machine_temp = build_bart_machine(training_X_k_red_by_vars_picked_by_method, 
                  training_y_k, num_burn_in = bart_machine$num_burn_in, 
                  num_iterations_after_burn_in = bart_machine$num_iterations_after_burn_in, 
                  num_trees = num_trees_pred_cv, run_in_sample = FALSE, 
                  verbose = FALSE)
                test_X_k_red_by_vars_picked_by_method = as.data.frame(test_X_k[, 
                  bart_variables_select_obj_k[[method]]])
                predict_obj = bart_predict_for_test_data(bart_machine_temp, 
                  test_X_k_red_by_vars_picked_by_method, text_y_k)
                destroy_bart_machine(bart_machine_temp)
                L2_err_mat[k, method] = predict_obj$L2_err
            }
        }
        cat("\n")
    }
    L2_err_by_method = colSums(L2_err_mat)
    min_var_selection_method = colnames(L2_err_mat)[which(L2_err_by_method == 
        min(L2_err_by_method))]
    min_var_selection_method = min_var_selection_method[1]
    cat("final", "\n")
    bart_variables_select_obj = var_selection_by_permute_response_three_methods(bart_machine, 
        num_permute_samples = num_permute_samples, num_trees_for_permute = num_trees_for_permute, 
        alpha = alpha, plot = FALSE)
    list(best_method = min_var_selection_method, important_vars_cv = sort(bart_variables_select_obj[[min_var_selection_method]]))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
