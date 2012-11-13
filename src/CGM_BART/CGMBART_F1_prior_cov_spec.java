package CGM_BART;

import gnu.trove.list.array.TIntArrayList;

public class CGMBART_F1_prior_cov_spec extends CGMBART_09_eval {
	private static final long serialVersionUID = -7765686625013501694L;

	protected boolean use_prior_cov_spec;
	/** This is the prior on which covs to split */
	protected double[] cov_split_prior;	

	
	public int pickRandomPredictorThatCanBeAssigned(CGMBARTTreeNode node){
		if (use_prior_cov_spec){
			TIntArrayList predictors = node.predictorsThatCouldBeUsedToSplitAtNode();
			//get probs of split prior based on predictors that can be used and weight it accordingly
			double[] weighted_cov_split_prior_subset = getWeightedCovSplitPriorSubset(predictors, cov_split_prior);
//			System.out.println("predictors: " + Tools.StringJoin(predictors));
//			System.out.println("cov_split_prior: " + Tools.StringJoin(cov_split_prior));
//			System.out.println("weighted_cov_split_prior_subset: " + Tools.StringJoin(weighted_cov_split_prior_subset));
			//choose predictor based on random prior value
			return StatToolbox.multinomial_sample(predictors, weighted_cov_split_prior_subset);
		}
		return super.pickRandomPredictorThatCanBeAssigned(node);
	}	
	
	/**
	 * Gets the total number of predictors that could be used for rules at this juncture
	 * @param node 
	 * @return
	 */
	public double pAdj(CGMBARTTreeNode node){
		if (use_prior_cov_spec){
			if (node.padj == null){
				node.padj = node.predictorsThatCouldBeUsedToSplitAtNode().size();
			}
			if (node.padj == 0){
				return 0;
			}
			if (node.isLeaf){
				return node.padj;
			}			
			//pull out weighted cov split prior subset vector
			TIntArrayList predictors = node.predictorsThatCouldBeUsedToSplitAtNode();
			//get probs of split prior based on predictors that can be used and weight it accordingly
			double[] weighted_cov_split_prior_subset = getWeightedCovSplitPriorSubset(predictors, cov_split_prior);	
			
//			System.out.println("predictors: " + Tools.StringJoin(predictors));
//			System.out.println("splitAttributeM: " + node.splitAttributeM);
			//find index inside predictor vector
			int index = CGMBARTTreeNode.BAD_FLAG_int;
			for (int i = 0; i < predictors.size(); i++){
				if (predictors.get(i) == node.splitAttributeM){
					index = i;
					break;
				}
			}
			
			//return inverse probability
			
			return 1 / weighted_cov_split_prior_subset[index];
		}
		return super.pAdj(node);
	}	
	
	private double[] getWeightedCovSplitPriorSubset(TIntArrayList predictors, double[] cov_split_prior) {
		double[] weighted_cov_split_prior_subset = new double[predictors.size()];
		for (int i = 0; i < cov_split_prior.length; i++){
			weighted_cov_split_prior_subset[i] = cov_split_prior[predictors.get(i)];
		}
		Tools.weight_arr_by_sum(weighted_cov_split_prior_subset);
		return weighted_cov_split_prior_subset;
	}	

	/**
	 * set the general version of the vector here
	 * 
	 * @param cov_split_prior
	 */
	public void setCovSplitPrior(double[] cov_split_prior) {
		this.cov_split_prior = cov_split_prior;
	}	
	
	public void useCovPriorCovSpec(){
		use_prior_cov_spec = true;
	}
	
}