package CGM_BART;


public class CGMBART_F2_heteroskedasticity extends CGMBART_F1_prior_cov_spec {
	private static final long serialVersionUID = -3069428133597923502L;

	protected boolean use_heteroskedasticity;
	
	/** the variance of the errors */
	protected double[][] gibbs_samples_of_sigsq_hetero;
	protected double[][] gibbs_samples_of_sigsq_hetero_after_burn_in;	
	
	
	private double calcLnLikRatioGrowF2(CGMBARTTreeNode grow_node) {
		double[] sigsqs = gibbs_samples_of_sigsq_hetero[gibbs_sample_num - 1];
				
		//we need sum_inv_sigsqs for the parent and both children
		//as well as weighted sum responses for the parent and both children
		double sum_inv_sigsq_parent = 0;
		double sum_responses_weighted_by_inv_sigsq_parent = 0;
		for (int i = 0; i < grow_node.n_eta; i++){
			int index = grow_node.indicies[i];
			double sigsq_i = sigsqs[index];
			sum_inv_sigsq_parent += 1 / sigsq_i;
			sum_responses_weighted_by_inv_sigsq_parent += grow_node.responses[i] / sigsq_i;
		}
		double sum_inv_sigsq_left = 0;
		double sum_responses_weighted_by_inv_sigsq_left = 0;
		for (int i = 0; i < grow_node.left.n_eta; i++){
			int index = grow_node.left.indicies[i];
			double sigsq_i = sigsqs[index];
			sum_inv_sigsq_left += 1 / sigsq_i;
			sum_responses_weighted_by_inv_sigsq_left += grow_node.left.responses[i] / sigsq_i;
		}
		double sum_inv_sigsq_right = 0;
		double sum_responses_weighted_by_inv_sigsq_right = 0;
		for (int i = 0; i < grow_node.right.n_eta; i++){
			int index = grow_node.right.indicies[i];
			double sigsq_i = sigsqs[index];
			sum_inv_sigsq_right += 1 / sigsq_i;
			sum_responses_weighted_by_inv_sigsq_right += grow_node.right.responses[i] / sigsq_i;
		}		
		
		double one_plus_hyper_sigsq_mu_times_sum_inv_sigsq_parent = 1 + hyper_sigsq_mu * sum_inv_sigsq_parent;
		double one_plus_hyper_sigsq_mu_times_sum_inv_sigsq_left = 1 + hyper_sigsq_mu * sum_inv_sigsq_left;
		double one_plus_hyper_sigsq_mu_times_sum_inv_sigsq_right = 1 + hyper_sigsq_mu * sum_inv_sigsq_right;
		
		double a = Math.log(one_plus_hyper_sigsq_mu_times_sum_inv_sigsq_parent);
		double b = Math.log(one_plus_hyper_sigsq_mu_times_sum_inv_sigsq_left);
		double c = Math.log(one_plus_hyper_sigsq_mu_times_sum_inv_sigsq_right);

		double d = Math.pow(sum_responses_weighted_by_inv_sigsq_left, 2) / one_plus_hyper_sigsq_mu_times_sum_inv_sigsq_left;
		double e = Math.pow(sum_responses_weighted_by_inv_sigsq_right, 2) / one_plus_hyper_sigsq_mu_times_sum_inv_sigsq_right;
		double f = Math.pow(sum_responses_weighted_by_inv_sigsq_parent, 2) / one_plus_hyper_sigsq_mu_times_sum_inv_sigsq_parent;
				
		return 0.5 * (a - b - c) + hyper_sigsq_mu / 2 * (d + e - f);
	}
	
	private void SampleSigsqF2(int sample_num, double[] es) {
//		double sigsq = drawSigsqFromPosterior(sample_num, es);
		for (int i = 0; i < n; i++){			
			double[] one_residual = {es[i]};
			double sigsq = drawSigsqFromPosterior(sample_num, one_residual);
			gibbs_samples_of_sigsq_hetero[sample_num][i] = sigsq;	
//			System.out.println("sample sigsq e_i = " + es[i] + " sigsq_i = " + sigsq);
		}
	
	}	
	
	private void SampleMusF2(int sample_num, CGMBARTTreeNode tree) {
		double[] current_sigsqs = gibbs_samples_of_sigsq_hetero[sample_num - 1];
		assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(tree, current_sigsqs);	
	}	
	
	protected void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(CGMBARTTreeNode node, double[] sigsqs) {
//		System.out.println("assignLeafValsUsingPosteriorMeanAndCurrentSigsq sigsq: " + sigsq);
		if (node.isLeaf){
			//update ypred
			double sum_sigsqs_leaf = 0;
			for (int index : node.indicies){
				sum_sigsqs_leaf += sigsqs[index];
			}
			double posterior_var = calcLeafPosteriorVarF2(node, sum_sigsqs_leaf);
			//draw from posterior distribution
			double posterior_mean = calcLeafPosteriorMeanF2(node, sum_sigsqs_leaf, posterior_var);
//			System.out.println("assignLeafVals n_k = " + node.n_eta + " sum_nk_sq = " + Math.pow(node.n_eta, 2) + " node = " + node.stringLocation(true));
//			System.out.println("assignLeafVals sum_sigsqs_leaf = " + sum_sigsqs_leaf + " posterior_mean = " + posterior_mean + " posterior_sigsq = " + posterior_var + " node.avg_response = " + node.avg_response_untransformed());
			node.y_pred = StatToolbox.sample_from_norm_dist(posterior_mean, posterior_var);
			if (node.y_pred == StatToolbox.ILLEGAL_FLAG){				
				node.y_pred = 0.0; //this could happen on an empty node
				System.err.println("ERROR assignLeafFINAL " + node.y_pred + " (sigsq = " + Tools.StringJoin(sigsqs) + ")");
			}
			//now update yhats
			node.updateYHatsWithPrediction();
//			System.out.println("assignLeafFINAL g = " + gibbs_sample_num + " y_hat = " + node.y_pred + " (sigsqs = " + Tools.StringJoin(sigsqs) + ")");
		}
		else {
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(node.left, sigsqs);
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(node.right, sigsqs);
		}
	}	
	
	private double calcLeafPosteriorMeanF2(CGMBARTTreeNode node, double sum_sigsqs, double posterior_var) {
		return (Math.pow(node.n_eta, 2) * node.avgResponse() / sum_sigsqs) * posterior_var;
	}

	private double calcLeafPosteriorVarF2(CGMBARTTreeNode node, double sum_sigsqs) {
		return 1 / (1 / hyper_sigsq_mu + Math.pow(node.n_eta, 2) / sum_sigsqs);
	}

	public void useHeteroskedasticity(){
		use_heteroskedasticity = true;
	}
	
	protected void InitGibbsSamplingData(){
		super.InitGibbsSamplingData();
		if (use_heteroskedasticity){
			gibbs_samples_of_sigsq_hetero = new double[num_gibbs_total_iterations + 1][n];	
			gibbs_samples_of_sigsq_hetero_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in][n];	
		}	
	}	
	
	private void InitizializeSigsqF2() {
		double[] initial_sigsqs = gibbs_samples_of_sigsq_hetero[0];
		for (int i = 0; i < n; i++){
			initial_sigsqs[i] = INITIAL_SIGSQ;
		}	
	}	
	
	
	
	/////////////nothing but scaffold code below, do not alter!
	
	
	protected void InitizializeSigsq() {
		if (use_heteroskedasticity){
			InitizializeSigsqF2();
		}
		else {
			super.InitizializeSigsq();
		}
	}

	protected void SampleMus(int sample_num, CGMBARTTreeNode tree) {
		if (use_heteroskedasticity){
			SampleMusF2(sample_num, tree);
		}
		else {
			super.SampleMus(sample_num, tree);
		}
	}	

	protected void SampleSigsq(int sample_num, double[] es) {
		if (use_heteroskedasticity){
			SampleSigsqF2(sample_num, es);
		}
		else {
			super.SampleSigsq(sample_num, es);
		}		
	}

	protected double calcLnLikRatioGrow(CGMBARTTreeNode grow_node) {
		if (use_heteroskedasticity){
			return calcLnLikRatioGrowF2(grow_node);
		}
		return super.calcLnLikRatioGrow(grow_node);
	}

	
}
