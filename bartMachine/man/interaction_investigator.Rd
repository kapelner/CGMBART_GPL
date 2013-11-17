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
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (bart_machine, plot = TRUE, num_replicates_for_avg = 5, 
    num_trees_bottleneck = 20, num_var_plot = 50, cut_bottom = NULL, 
    bottom_margin = 10) 
{
    interaction_counts = array(NA, c(bart_machine$p, bart_machine$p, 
        num_replicates_for_avg))
    for (r in 1:num_replicates_for_avg) {
        if (r == 1 & num_trees_bottleneck == bart_machine$num_trees) {
            interaction_counts[, , r] = sapply(.jcall(bart_machine$java_bart_machine, 
                "[[I", "getInteractionCounts", as.integer(BART_NUM_CORES)), 
                .jevalArray)
        }
        else {
            bart_machine_dup = bart_machine_duplicate(bart_machine, 
                num_trees = num_trees_bottleneck)
            interaction_counts[, , r] = sapply(.jcall(bart_machine_dup$java_bart_machine, 
                "[[I", "getInteractionCounts", as.integer(BART_NUM_CORES)), 
                .jevalArray)
            destroy_bart_machine(bart_machine_dup)
            cat(".")
            if (r\%\%40 == 0) {
                cat("\n")
            }
        }
    }
    cat("\n")
    interaction_counts_avg = apply(interaction_counts, 1:2, mean)
    if (bart_machine$use_missing_data == T) {
        rownames(interaction_counts_avg) = bart_machine$training_data_features_with_missing_features
        colnames(interaction_counts_avg) = bart_machine$training_data_features_with_missing_features
    }
    else {
        rownames(interaction_counts_avg) = bart_machine$training_data_features
        colnames(interaction_counts_avg) = bart_machine$training_data_features
    }
    interaction_counts_sd = apply(interaction_counts, 1:2, sd)
    interaction_counts_s_w_test = apply(interaction_counts, 1:2, 
        shapiro_wilk_p_val)
    avg_counts = array(NA, bart_machine$p * (bart_machine$p - 
        1)/2)
    sd_counts = array(NA, bart_machine$p * (bart_machine$p - 
        1)/2)
    iter = 1
    for (i in 1:bart_machine$p) {
        for (j in 1:bart_machine$p) {
            if (j <= i) {
                avg_counts[iter] = interaction_counts_avg[i, 
                  j]
                sd_counts[iter] = interaction_counts_sd[i, j]
                names(avg_counts)[iter] = paste(rownames(interaction_counts_avg)[i], 
                  "x", rownames(interaction_counts_avg)[j])
                iter = iter + 1
            }
        }
    }
    num_total_interactions = bart_machine$p * (bart_machine$p + 
        1)/2
    if (num_var_plot == Inf || num_var_plot > num_total_interactions) {
        num_var_plot = num_total_interactions
    }
    avg_counts_sorted_indices = sort(avg_counts, decreasing = TRUE, 
        index.return = TRUE)$ix
    avg_counts = avg_counts[avg_counts_sorted_indices][1:num_var_plot]
    sd_counts = sd_counts[avg_counts_sorted_indices][1:num_var_plot]
    if (is.null(cut_bottom)) {
        ylim_bottom = 0
    }
    else {
        ylim_bottom = cut_bottom * min(avg_counts)
    }
    if (plot) {
        par(mar = c(bottom_margin, 6, 3, 0))
        if (is.na(sd_counts[1])) {
            moe = 0
        }
        else {
            moe = 1.96 * sd_counts/sqrt(num_replicates_for_avg)
        }
        bars = barplot(avg_counts, names.arg = names(avg_counts), 
            las = 2, ylab = "Relative Importance", col = "gray", 
            ylim = c(ylim_bottom, max(avg_counts + moe)), xpd = FALSE)
        if (!is.na(sd_counts[1])) {
            conf_upper = avg_counts + 1.96 * sd_counts/sqrt(num_replicates_for_avg)
            conf_lower = avg_counts - 1.96 * sd_counts/sqrt(num_replicates_for_avg)
            segments(bars, avg_counts, bars, conf_upper, col = rgb(0.59, 
                0.39, 0.39), lwd = 3)
            segments(bars, avg_counts, bars, conf_lower, col = rgb(0.59, 
                0.39, 0.39), lwd = 3)
        }
        par(mar = c(5.1, 4.1, 4.1, 2.1))
    }
    invisible(list(interaction_counts_avg = interaction_counts_avg, 
        interaction_counts_sd = interaction_counts_sd, interaction_counts_s_w_test = interaction_counts_s_w_test))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
