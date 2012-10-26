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
//			System.out.println("assign Leaf y_pred: " + un_transform_y(node.y_pred) + " (sigsq = " + sigsq * y_range_sq + ")");
			node.updateYHatsWithPrediction();
			
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
//		System.out.println("calcLeafPosteriorVar: node.n_eta = " + node.nEta());
		return 1 / (1 / hyper_sigsq_mu + node.n_eta / sigsq);
	}
	
	protected double drawSigsqFromPosterior(int sample_num, double[] R_j) {
		//first calculate the SSE
		
		double[] es = getResidualsFromFullSumModel(sample_num, R_j);
//		System.out.println("drawSigsqFromPosterior es = " + Tools.StringJoin(es, ",")); 
//		if (WRITE_DETAILED_DEBUG_FILES){		
//			remainings.println((sample_num) + ",,e," + Tools.StringJoin(es, ","));
//			evaluations.println((sample_num) + ",,e," + Tools.StringJoin(es, ","));
//		}
		double sse = 0;
		for (double e : es){
			sse += Math.pow(e, 2); 
		}
		//we're sampling from sigsq ~ InvGamma((nu + n) / 2, 1/2 * (sum_i error^2_i + lambda * nu))
		//which is equivalent to sampling (1 / sigsq) ~ Gamma((nu + n) / 2, 2 / (sum_i error^2_i + lambda * nu))
		return StatToolbox.sample_from_inv_gamma((hyper_nu + n) / 2, 2 / (sse + hyper_nu * hyper_lambda));
	}
	
	protected double[] getResidualsFromFullSumModel(int sample_num, double[] R_j){	
		//all we need to do is subtract the last tree's yhats now
		CGMBARTTreeNode last_tree = gibbs_samples_of_cgm_trees.get(sample_num).get(num_trees - 1);
		for (int i = 0; i < n; i++){
			R_j[i] -= last_tree.yhats[i];
		}
		return R_j;
	}	
}
