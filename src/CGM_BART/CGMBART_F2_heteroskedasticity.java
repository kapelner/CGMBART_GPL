package CGM_BART;

import java.util.ArrayList;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;


public class CGMBART_F2_heteroskedasticity extends CGMBART_F1_prior_cov_spec {
	private static final long serialVersionUID = -3069428133597923502L;

	protected boolean use_heteroskedasticity;
	
	protected double hyper_q_sigsq = 0.9;
	protected double hyper_nu_sigsq = 3.0;
	protected double hyper_lambda_sigsq;
	protected double sample_var_e;
	
	/** the variance of the errors as well as other things necessary for Gibbs sampling */
	protected double[][] gibbs_samples_of_sigsq_hetero;
	protected double[][] gibbs_samples_of_sigsq_hetero_after_burn_in;	
	protected double[][] gibbs_samples_of_betas_for_lm_sigsqs;
	protected double[][] gibbs_samples_of_betas_for_lm_sigsqs_after_burn_in;
	protected double[] gibbs_samples_of_tausq_for_lm_sigsqs;
	protected double[] gibbs_samples_of_tausq_for_lm_sigsqs_after_burn_in;

	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		tabulateSimulationDistributionsF2();
		calculateHyperparametersF2();		
	}
	
	protected void tabulateSimulationDistributionsF2() {
		StatToolbox.cacheInvGammas(hyper_nu_sigsq, n, this);
	}
	
	private void calculateHyperparametersF2() {
		super.calculateHyperparameters();
		double ten_pctile_chisq_df_hyper_nu = 0;		
		ChiSquaredDistributionImpl chi_sq_dist = new ChiSquaredDistributionImpl(hyper_nu_sigsq);
		try {
			ten_pctile_chisq_df_hyper_nu = chi_sq_dist.inverseCumulativeProbability(1 - hyper_q_sigsq);
		} catch (MathException e) {
			System.err.println("Could not calculate inverse cum prob density for chi sq df = " + hyper_nu_sigsq + " with q = " + hyper_q_sigsq);
			System.exit(0);
		}

		hyper_lambda_sigsq = ten_pctile_chisq_df_hyper_nu / hyper_nu_sigsq * sample_var_e;		
	}
	
	
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
		//this comes in three steps
		SampleBetaForLMSigsqs(es);
		SampleTausqForLMSigsqs(es);
		SampleSigsqsViaLM(es, sample_num);
	}
	
	private void SampleSigsqsViaLM(double[] es, int sample_num) {
		double[] beta_lm_sigsq = gibbs_samples_of_betas_for_lm_sigsqs[sample_num];
		double tausq_lm_sigsq = gibbs_samples_of_tausq_for_lm_sigsqs[sample_num];
		double[] sigsqs_gibbs_sample = gibbs_samples_of_sigsq_hetero[sample_num];
		for (int i = 0; i < n; i++){
			//initialize to be the intercept
			double x_trans_beta_lm_sigsq = beta_lm_sigsq[0];
			for (int j = 1; j <= p; j++){
				x_trans_beta_lm_sigsq += X_y.get(i)[j - 1] * beta_lm_sigsq[j];
			}
			sigsqs_gibbs_sample[i] = StatToolbox.sample_from_norm_dist(x_trans_beta_lm_sigsq, tausq_lm_sigsq);
		}
	}

	private void SampleTausqForLMSigsqs(double[] es) {
		double sse = 0;
		for (double e : es){
			sse += e * e; 
		}
//		System.out.println("hyper_nu = " + hyper_nu + " hyper_lambda = " + hyper_lambda + " esl = " + es.length + " sse = " + sse);
		//we're sampling from sigsq ~ InvGamma((nu + n) / 2, 1/2 * (sum_i error^2_i + lambda * nu))
		//which is equivalent to sampling (1 / sigsq) ~ Gamma((nu + n) / 2, 2 / (sum_i error^2_i + lambda * nu))
		return StatToolbox.sample_from_inv_gamma((hyper_nu_sigsq + n + p + 1) / 2, 2 / (sse + hyper_nu * hyper_lambda), this);
	}

	private void SampleBetaForLMSigsqs(double[] es) {
		// TODO Auto-generated method stub
		
	}

	public double[] getSigsqsByGibbsSample(int g){
		return gibbs_samples_of_sigsq_hetero[g];
	}
	
	private void SampleMusF2(int sample_num, CGMBARTTreeNode tree) {
		double[] current_sigsqs = gibbs_samples_of_sigsq_hetero[sample_num - 1];
		assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(tree, current_sigsqs);	
	}	
	
	protected void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(CGMBARTTreeNode node, double[] sigsqs) {
//		System.out.println("assignLeafValsUsingPosteriorMeanAndCurrentSigsq sigsq: " + sigsq);
		if (node.isLeaf){
			//update ypred
			double posterior_var = calcLeafPosteriorVarF2(node, sigsqs);
			//draw from posterior distribution
			double posterior_mean = calcLeafPosteriorMeanF2(node, posterior_var, sigsqs);
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
	
	private double calcLeafPosteriorMeanF2(CGMBARTTreeNode node, double posterior_var, double[] sigsqs) {
		double numerator = 0;
		for (int ell = 0; ell < node.n_eta; ell++){
			numerator += node.responses[ell] / sigsqs[node.indicies[ell]];
		}		
		return numerator * posterior_var;
	}

	private double calcLeafPosteriorVarF2(CGMBARTTreeNode node, double[] sigsqs) {
		double sum_sigsqs_leaf = 0;
		for (int index : node.indicies){
			sum_sigsqs_leaf += sigsqs[index];
		}
		return 1 / (1 / hyper_sigsq_mu + 1 / sum_sigsqs_leaf);
	}

	public void useHeteroskedasticity(){
		use_heteroskedasticity = true;
	}
	
	protected void InitGibbsSamplingData(){
		super.InitGibbsSamplingData();
		if (use_heteroskedasticity){
			gibbs_samples_of_sigsq_hetero = new double[num_gibbs_total_iterations + 1][n];	
			gibbs_samples_of_sigsq_hetero_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in][n];
			gibbs_samples_of_betas_for_lm_sigsqs = new double[num_gibbs_total_iterations + 1 ][p + 1];
			gibbs_samples_of_betas_for_lm_sigsqs_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in][p + 1];
			gibbs_samples_of_tausq_for_lm_sigsqs = new double[num_gibbs_total_iterations + 1];
			gibbs_samples_of_tausq_for_lm_sigsqs_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in];
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

	public void setQSigsq(double hyper_q_sigsq) {
		this.hyper_q_sigsq = hyper_q_sigsq;
	}

	public void setNuSigsq(double hyper_nu_sigsq) {
		this.hyper_nu_sigsq = hyper_nu_sigsq;
	}
	
	public void setSampleVarResiduals(double sample_var_e){
		this.sample_var_e = sample_var_e;
	}
	
}
