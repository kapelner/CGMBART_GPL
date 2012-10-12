package CGM_BART;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


public abstract class CGMBART_06_gibbs_internal extends CGMBART_05_gibbs_base implements Serializable {
	private static final long serialVersionUID = 5591873635969255497L;

	/** who are the previous trees?
	 * e.g. if t=0, then we take all 1, ..., m-1 trees from previous gibbs sample
	 *     if t=1, then we take the 0th tree from this gibbs sample, and 2, ..., m-1 trees from the previous gibbs sample
	 *     ...
	 *     if t=j, then we take the 0,...,j-1 trees from this gibbs sample, and j+1, ..., m-1 trees from the previous gibbs sample
	 *     ...
	 *    if t=m-1, then we take all 0, ..., m-2 trees from this gibbs sample
	 * so let's put together this list of trees:	
	**/ 
	protected ArrayList<CGMBARTTreeNode> findOtherTrees(int sample_num, int t){
		ArrayList<CGMBARTTreeNode> other_trees = new ArrayList<CGMBARTTreeNode>(num_trees - 1);
		for (int j = 0; j < t; j++){
			other_trees.add(gibbs_samples_of_cgm_trees.get(sample_num).get(j));
		}
		for (int j = t + 1; j < num_trees; j++){
			other_trees.add(gibbs_samples_of_cgm_trees.get(sample_num - 1).get(j));
		}
		return other_trees;
	}
	
	protected double[] getResidualsBySubtractingTrees(List<CGMBARTTreeNode> trees_to_subtract) {
		double[] sum_ys_without_jth_tree = new double[n];
		double[] Rjs = new double[n];
		
		for (int i = 0; i < n; i++){
			sum_ys_without_jth_tree[i] = 0; //initialize at zero, then add it up over all trees except the jth
			for (int t = 0; t < trees_to_subtract.size(); t++){
//				double y_i = un_transform_y(trees_to_subtract.get(t).Evaluate(X_y.get(i)));
				double y_i = trees_to_subtract.get(t).Evaluate(X_y.get(i));
				sum_ys_without_jth_tree[i] += y_i;
//				System.out.println(y_i);
			}
			//now we need to subtract this from y
			Rjs[i] = y_trans[i] - sum_ys_without_jth_tree[i];
		}
//		System.out.println("getResidualsForAllTreesExcept one " +  Tools.StringJoin(Rjs, ", "));
		return Rjs;
	}
	
	protected void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(CGMBARTTreeNode node, double sigsq) {
//		System.out.println("assignLeafValsUsingPosteriorMeanAndCurrentSigsq sigsq: " + sigsq);
		if (node.isLeaf){
			double posterior_sigsq = calcLeafPosteriorVar(node, sigsq);
			//draw from posterior distribution
			double posterior_mean = calcLeafPosteriorMean(node, sigsq, posterior_sigsq);
//			System.out.println("posterior_mean = " + posterior_mean + " node.avg_response = " + node.avg_response_untransformed());
			node.y_prediction = StatToolbox.sample_from_norm_dist(posterior_mean, posterior_sigsq);
			if (node.y_prediction == StatToolbox.ILLEGAL_FLAG){				
				node.y_prediction = 0.0; //this could happen on an empty node
				System.err.println("ERROR assignLeafFINAL " + node.y_prediction + " (sigsq = " + sigsq + ")");
			}
//			System.out.println("assignLeafFINAL " + un_transform_y(node.y_prediction) + " (sigsq = " + sigsq * y_range_sq + ")");
		}
		else {
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(node.left, sigsq);
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(node.right, sigsq);
		}
	}

	protected double calcLeafPosteriorMean(CGMBARTTreeNode node, double sigsq, double posterior_var) {
//		System.out.println("leafPosteriorMean hyper_sigsq_mu " + hyper_sigsq_mu + " node.n " + node.n + " sigsq " + sigsq + " node.avg_response() " + node.avg_response() + " posterior_var " + posterior_var);
		return (hyper_mu_mu / hyper_sigsq_mu + node.n_eta / sigsq * node.avgResponse()) / (1 / posterior_var);
	}

	protected double calcLeafPosteriorVar(CGMBARTTreeNode node, double sigsq) {
//		System.out.println("leafPosteriorVar sigsq " + sigsq + " var " + 1 / (1 / hyper_sigsq_mu + node.n * m / sigsq));
		return 1 / (1 / hyper_sigsq_mu + node.n_eta / sigsq);
	}
	
	protected double drawSigsqFromPosterior(int sample_num) {
		//first calculate the SSE
		double sum_sq_errors = 0;
		double[] es = getErrorsForAllTrees(sample_num);
		if (WRITE_DETAILED_DEBUG_FILES){		
			remainings.println((sample_num) + ",,e," + Tools.StringJoin(es, ","));
			evaluations.println((sample_num) + ",,e," + Tools.StringJoin(es, ","));
		}
		for (double e : es){
			sum_sq_errors += Math.pow(e, 2); 
		}
//		System.out.println("sample: " + sample_num + " sse = " + sum_sq_errors);
		//we're sampling from sigsq ~ InvGamma((nu + n) / 2, 1/2 * (sum_i error^2_i + lambda * nu))
		//which is equivalent to sampling (1 / sigsq) ~ Gamma((nu + n) / 2, 2 / (sum_i error^2_i + lambda * nu))
		double sigsq = StatToolbox.sample_from_inv_gamma((hyper_nu + n) / 2, 2 / (sum_sq_errors + hyper_nu * hyper_lambda));
//		System.out.println("\n\nSample posterior from trees " + treeIDsInCurrentSample(sample_num) + "  SSE=" + sum_sq_errors + " post_sigsq_sample: " + sigsq);
//		
//		//debug
//		double[] posterior_sigma_simus = new double[1000];
//		for (int i = 0; i < 1000; i++){
//			posterior_sigma_simus[i] = StatToolbox.sample_from_inv_gamma((hyper_nu + n) / 2, 2 / (sum_sq_errors + hyper_nu * hyper_lambda)) * (TRANSFORM_Y ? y_range_sq : 1);
//		}
//		System.out.print("\n\n\n");
//		sigsqs_draws.println(sample_num + "," + hyper_nu + "," + hyper_lambda + "," + n + "," + sum_sq_errors + "," + (sigsq * (TRANSFORM_Y ? y_range_sq : 1)) + "," + y_range_sq + "," + IOTools.StringJoin(posterior_sigma_simus, ","));
		
		return sigsq;
	}
	
	protected double[] getErrorsForAllTrees(int sample_num){
//		System.out.println("getErrorsForAllTrees");
		double[] sum_ys_trans = new double[n];
		for (int i = 0; i < n; i++){
			sum_ys_trans[i] = 0; //initialize at zero, then add it up over all trees except the jth
			for (int t = 0; t < num_trees; t++){
//				System.out.println("getErrorsForAllTrees m = " + m);
				//obviously y_vec - \sum_i g_i = \sum_i y_i - g_i
				CGMBARTTreeNode tree = gibbs_samples_of_cgm_trees.get(sample_num).get(t);
				double y_hat_trans = tree.Evaluate(X_y.get(i));
//				double y_hat = un_transform_y(tree.Evaluate(X_y.get(i)));
//				System.out.println("i = " + (i + 1) + " y: " + y[i] + " y_hat: " + y_hat + " e: " + (y[i] - y_hat)+ " tree " + t);
				sum_ys_trans[i] += y_hat_trans;
			}
		}
		//now we need to subtract this from y
		double[] error_js = new double[n];
		for (int i = 0; i < n; i++){
			error_js[i] = y_trans[i] - sum_ys_trans[i];
//			errorjs[i] = transform_y(errorjs[i]);
		}
//		System.out.println("sum_ys " + IOTools.StringJoin(sum_ys, ","));
//		System.out.println("y_trans " + IOTools.StringJoin(y_trans, ","));
//		System.out.println("errorjs " + IOTools.StringJoin(errorjs, ","));
		return error_js;
	}	
}
