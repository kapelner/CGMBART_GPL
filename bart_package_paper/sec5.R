library(bartMachine)
library(MASS)

data(Pima.te)
X = Pima.te[, -8]
y = Pima.te[, 8]

set_bart_machine_num_cores(4)
init_java_for_bart_machine_with_mem_in_mb(2500)


bart_machine = build_bart_machine(X, y)
y_hat = predict(bart_machine, X)

#k_fold_cv(X, y)
#doesn't work
bart_machine_cv = build_bart_machine_cv(X, y)
bart_machine_cv

bart_machine = build_bart_machine(X, y, prob_rule_class = 0.3)
bart_machine

#doesn't work
plot_y_vs_yhat(bart_machine_cv, credible_intervals = TRUE)
plot_y_vs_yhat(bart_machine_cv, prediction_intervals = TRUE)


oos_stats = k_fold_cv(X, y, k_folds = 10)
oos_stats$confusion_matrix


get_var_counts_over_chain(bart_machine_cv)

get_var_props_over_chain(bart_machine_cv)

cov_importance_test(bart_machine_cv, covariates = c("age"))

check_bart_error_assumptions(bart_machine_cv)
hist_sigsqs(bart_machine_cv)
get_sigsqs(bart_machine_cv)

plot_sigsqs_convergence_diagnostics(bart_machine_cv)

investigate_var_importance(bart_machine_cv)

plot_convergence_diagnostics(bart_machine_cv)

interaction_investigator(bart_machine_cv)

pd_plot(bart_machine_cv, j = "glu")

plot_tree_depths(bart_machine_cv)
get_tree_depths(bart_machine_cv)
plot_tree_num_nodes(bart_machine_cv)
plot_mh_acceptance_reject(bart_machine_cv)

rmse_by_num_trees(bart_machine_cv)

bart_predict_for_test_data(bart_machine_cv, X[1 : 50, ], y[1 : 50])

bart_machine_get_posterior(bart_machine_cv, X[1 : 5, ])
calc_credible_intervals(bart_machine_cv, X[1 : 2, ])
calc_prediction_intervals(bart_machine_cv, X[1 : 2, ])

vs = var_selection_by_permute_response_three_methods(bart_machine_cv, bottom_margin = 10, num_permute_samples=10) ##fix dot printing
names(vs)
cv_vars = var_selection_by_permute_response_cv(bart_machine_cv, k_folds = 2, num_reps_for_avg = 5, num_permute_samples = 20, num_trees_pred_cv = 50)





