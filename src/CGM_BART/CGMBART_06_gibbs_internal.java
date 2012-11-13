package CGM_BART;

import gnu.trove.list.array.TIntArrayList;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


public abstract class CGMBART_06_gibbs_internal extends CGMBART_05_gibbs_base implements Serializable {
	private static final long serialVersionUID = 5591873635969255497L;

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
			node.updateYHatsWithPrediction();
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
	

	/**
	 * Pick a random predictor from the set of possible predictors at this juncture
	 * @param node 
	 * @return
	 */
	public int pickRandomPredictorThatCanBeAssigned(CGMBARTTreeNode node){
        TIntArrayList predictors = node.predictorsThatCouldBeUsedToSplitAtNode();
        return predictors.get((int)Math.floor(StatToolbox.rand() * pAdj(node)));
	}
	
	
	
	/**
	 * Gets the total number of predictors that could be used for rules at this juncture
	 * @param node 
	 * @return
	 */
	public double pAdj(CGMBARTTreeNode node){
		if (node.padj == null){
			node.padj = node.predictorsThatCouldBeUsedToSplitAtNode().size();
		}
		return node.padj;
	}	
}
