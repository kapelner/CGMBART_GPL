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
//		double[] sum_ys_without_jth_tree = new double[n];
		double[] Rjs = new double[n];
		
		//initialize Rjs to be y
		for (int i = 0; i < n; i++){
			Rjs[i] = y_trans[i];
		}
		
		//now go through and get the yhats for each tree and subtract them from Rjs
		for (CGMBARTTreeNode tree : trees_to_subtract){
			double[] y_hat_vec = tree.yhats;
			//subtract them from Rjs
			for (int i = 0; i < n; i++){
				Rjs[i] -= y_hat_vec[i];
			}			
		}
		
//		for (int i = 0; i < n; i++){
//			sum_ys_without_jth_tree[i] = 0; //initialize at zero, then add it up over all trees except the jth
//			for (CGMBARTTreeNode tree : trees_to_subtract){
////				double y_i = un_transform_y(trees_to_subtract.get(t).Evaluate(X_y.get(i)));
//				double y_i = tree.Evaluate(X_y.get(i));
//				sum_ys_without_jth_tree[i] += y_i;
////				System.out.println(y_i);
//			}
//			//now we need to subtract this from y
//			Rjs[i] = y_trans[i] - sum_ys_without_jth_tree[i];
//		}
//		System.out.println("getResidualsForAllTreesExcept one " +  Tools.StringJoin(Rjs, ", "));
		return Rjs;
	}
	
	protected void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqAndUpdateYhats(CGMBARTTreeNode node, double sigsq) {
//		System.out.println("assignLeafValsUsingPosteriorMeanAndCurrentSigsq sigsq: " + sigsq);
		if (node.isLeaf){
			//update ypred
			double posterior_var = calcLeafPosteriorVar(node, sigsq);
			//draw from posterior distribution
			double posterior_mean = calcLeafPosteriorMean(node, sigsq, posterior_var);
//			System.out.println("assignLeafVals posterior_mean = " + posterior_mean + " posterior_sigsq = " + posterior_var + " node.avg_response = " + node.avg_response_untransformed());
			node.y_pred = StatToolbox.sample_from_norm_dist(posterior_mean, posterior_var);
			if (node.y_pred == StatToolbox.ILLEGAL_FLAG){				
				node.y_pred = 0.0; //this could happen on an empty node
				System.err.println("ERROR assignLeafFINAL " + node.y_pred + " (sigsq = " + sigsq + ")");
			}
			//now update yhats
			for (int index : node.getIndices()){
				node.yhats[index] = node.y_pred;
			}
//			System.out.println("assignLeafFINAL " + un_transform_y(node.y_prediction) + " (sigsq = " + sigsq * y_range_sq + ")");
		}
		else {
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqAndUpdateYhats(node.left, sigsq);
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqAndUpdateYhats(node.right, sigsq);
		}
	}

	protected double calcLeafPosteriorMean(CGMBARTTreeNode node, double sigsq, double posterior_var) {
//		System.out.println("leafPosteriorMean hyper_sigsq_mu " + hyper_sigsq_mu + " node.n " + node.n + " sigsq " + sigsq + " node.avg_response() " + node.avg_response() + " posterior_var " + posterior_var);
		return (hyper_mu_mu / hyper_sigsq_mu + node.n_eta / sigsq * node.avgResponse()) / (1 / posterior_var);
	}

	protected double calcLeafPosteriorVar(CGMBARTTreeNode node, double sigsq) {
//		System.out.println("calcLeafPosteriorVar: node.n_eta = " + node.n_eta);
		return 1 / (1 / hyper_sigsq_mu + node.n_eta / sigsq);
	}
	
	protected double drawSigsqFromPosterior(int sample_num, double[] residual_vec_excluding_last_tree) {
		//first calculate the SSE
		double sse = 0;
		double[] es = getResidualsFromFullSumModel(sample_num, residual_vec_excluding_last_tree);
		if (WRITE_DETAILED_DEBUG_FILES){		
			remainings.println((sample_num) + ",,e," + Tools.StringJoin(es, ","));
			evaluations.println((sample_num) + ",,e," + Tools.StringJoin(es, ","));
		}
		for (double e : es){
			sse += Math.pow(e, 2); 
		}
//		System.out.println("sample: " + sample_num + " sse = " + sum_sq_errors);
		//we're sampling from sigsq ~ InvGamma((nu + n) / 2, 1/2 * (sum_i error^2_i + lambda * nu))
		//which is equivalent to sampling (1 / sigsq) ~ Gamma((nu + n) / 2, 2 / (sum_i error^2_i + lambda * nu))
		double sigsq = StatToolbox.sample_from_inv_gamma((hyper_nu + n) / 2, 2 / (sse + hyper_nu * hyper_lambda));
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
	
	protected double[] getResidualsFromFullSumModel(int sample_num, double[] residual_vec_excluding_last_tree){
		double[] residuals = new double[n];
		
		CGMBARTTreeNode last_tree = gibbs_samples_of_cgm_trees.get(sample_num).get(num_trees - 1);
		double[] residual_last_tree = last_tree.yhats;
		for (int i = 0; i < n; i++){
			residuals[i] = y_trans[i] - residual_vec_excluding_last_tree[i] - residual_last_tree[i];
		}
		return residuals;
	}	
}
