package CGM_BART;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;

import Jama.Matrix;
import Jama.QRDecomposition;


public class CGMBART_F2_heteroskedasticity extends CGMBART_F1_prior_cov_spec {
	private static final long serialVersionUID = -3069428133597923502L;

	private static final double IntialTauSqLM = 1;
	
	protected boolean use_heteroskedasticity = false;
	
	protected double hyper_q_sigsq = 0.9;
	protected double hyper_nu_sigsq = 3.0;
	protected double hyper_lambda_sigsq;
	protected double sample_var_e = 11.20568;
	protected double hyper_beta_sigsq = 10;
	
	/** the variance of the errors as well as other things necessary for Gibbs sampling */
	protected double[][] gibbs_samples_of_sigsq_hetero;
	protected double[][] gibbs_samples_of_sigsq_hetero_after_burn_in;	
	protected double[][] gibbs_samples_of_betas_for_lm_sigsqs;
	protected double[][] gibbs_samples_of_betas_for_lm_sigsqs_after_burn_in;
	protected double[] gibbs_samples_of_tausq_for_lm_sigsqs;
	protected double[] gibbs_samples_of_tausq_for_lm_sigsqs_after_burn_in;

	private Matrix Xmat_star;

	private double sigsq_from_vanilla_bart;

	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		if (use_heteroskedasticity){
			System.out.println("n: " + n + " p: " + p);
			
			tabulateSimulationDistributionsF2();
			calculateHyperparametersF2();
			
			//precompute X as a Matrix object
			Xmat_star = new Matrix(n + p + 1, p + 1);
			//the top part is just the original X training matrix
			for (int i = 0; i < n; i++){
				for (int j = 0; j < p + 1; j++){
					if (j == 0){
						Xmat_star.set(i, j, 1); //this is the intercept
					}
					else {
						Xmat_star.set(i, j, X_y.get(i)[j - 1]);
					}
				}
			}
			//the bottom portion is the identity matrix
			for (int i = n; i < n + p + 1; i++){
				Xmat_star.set(i, i - n, 1);
			}
			
//			System.out.println("Xmat_star");
//			Xmat_star.print(3, 5);
		}
	}

	protected void tabulateSimulationDistributionsF2() {
		StatToolbox.cacheInvGammas(hyper_nu_sigsq, n, this);
	}
	
	private void calculateHyperparametersF2() {
		double ten_pctile_chisq_df_hyper_nu = 
			new ChiSquaredDistribution(hyper_nu_sigsq).inverseCumulativeProbability(1 - hyper_q_sigsq);

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
	
	private double calcLnLikRatioChangeF2(CGMBARTTreeNode eta, CGMBARTTreeNode eta_star) {
		//TODO
		return 0;
	}		
	
	private void SampleSigsqF2(int sample_num, double[] es) {
		System.out.println("\n\nGibbs sample_num: " + sample_num + "  Sigsqs \n" + "----------------------------------------------------");
//		System.out.println("es: " + Tools.StringJoin(es));
//		
//		//first convert the residuals to log residual squareds
//		//first calculate the "epsilon" vector
//		Matrix log_sq_resid_vec = new Matrix(n + p + 1, 1);
//		
//		for (int i = 0; i < n; i++){
//			log_sq_resid_vec.set(i, 0, Math.log(Math.pow(es[i], 2)));
//		}
//		for (int i = n; i < n + p + 1; i++){
//			log_sq_resid_vec.set(i, 0, 0);
//		}
//		
//		System.out.println("log_sq_resid_vec");
//		log_sq_resid_vec.transpose().print(3, 5);
//		
//		
//		////this comes in three steps
//		
//		//1 - draw beta
//		HashMap<String, Matrix> beta_sample = SampleBetaForLMSigsqs(log_sq_resid_vec, gibbs_samples_of_tausq_for_lm_sigsqs[sample_num - 1], sample_num);
//		Matrix beta_draw_matrix = beta_sample.get("beta_draw");
//		Matrix beta_vec = beta_sample.get("beta_vec");
//		
//		System.out.println("beta_draw_matrix");
//		beta_draw_matrix.print(3, 5);
//		System.out.println("beta_vec");
//		beta_vec.print(3, 5);
//		
//		
//		//now take this matrix form and convert it to a double vec which is a pain in the neck
//		double[][] beta_draw_double_matrix = beta_draw_matrix.getArray();
//		double[] beta_draw = new double[p + 1];
//		for (int i = 0; i < p + 1; i++){
//			beta_draw[i] = beta_draw_double_matrix[i][0];
//		}
//		
//		double[][] beta_vec_double_matrix = beta_vec.getArray();
//		double[] beta_vec_vec = new double[p + 1];
//		for (int i = 0; i < p + 1; i++){
//			beta_vec_vec[i] = beta_vec_double_matrix[i][0];
//		}		
//		
//		gibbs_samples_of_betas_for_lm_sigsqs[sample_num] = beta_vec_vec; //SSSSSSSSSSSSSSSSSSSSSS
//		System.out.println("beta_draw_vec: " + Tools.StringJoin(gibbs_samples_of_betas_for_lm_sigsqs[sample_num]));
//		
//		//now calculate the y_hats from using the recently sampled beta
//		Matrix y_hat = Xmat_star.times(beta_vec);
//		Matrix resids = log_sq_resid_vec.minus(y_hat);
//		
//		//2 - draw tausq
//		SampleTausqForLMSigsqs(resids, sample_num);
		//3 - draw sigsqs and send back to BART
		super.SampleSigsq(sample_num, es);
		SampleSigsqsViaLM(sample_num);
	}

	private HashMap<String, Matrix> SampleBetaForLMSigsqs(Matrix log_sq_resid_vec, double tau_sq, int sample_num) {
		
		Matrix Sigma_star_neg_half = new Matrix(n + p + 1, n + p + 1);
		double one_over_tau = 1 / Math.sqrt(tau_sq);
		double one_over_sqrt_hyper_beta_sigsq = 1 / Math.sqrt(hyper_beta_sigsq);
		for (int i = 0; i < n; i++){
			Sigma_star_neg_half.set(i, i, one_over_tau);
		}
		for (int i = n; i < n + p + 1; i++){
			Sigma_star_neg_half.set(i, i, one_over_sqrt_hyper_beta_sigsq);
		}
		
//		System.out.println("Sigma_star_neg_half");
//		Sigma_star_neg_half.print(3, 5);
		
		Matrix Sigma_star_neg_half_X_mat = Sigma_star_neg_half.times(Xmat_star);
		Matrix Sigma_star_neg_half_log_sq_resid_vec = Sigma_star_neg_half.times(log_sq_resid_vec);
		
//		System.out.println("Sigma_star_neg_half_X_mat");
//		Sigma_star_neg_half_X_mat.print(3, 5);
//		System.out.println("Sigma_star_neg_half_log_sq_resid_vec");
//		Sigma_star_neg_half_log_sq_resid_vec.print(3, 5);
		
		QRDecomposition QR = new QRDecomposition(Sigma_star_neg_half_X_mat);
		Matrix Qt = QR.getQ().transpose();
		Matrix R = QR.getR();
//		System.out.println("R");
//		R.print(3, 5);		
		Matrix Rinv = R.inverse();
//		System.out.println("Qt");
//		Qt.print(3, 5);
		System.out.println("Rinv");
		Rinv.print(3, 5);
		Matrix V_beta = Rinv.times(Rinv.transpose());
		Matrix Beta_vec = Rinv.times(Qt).times(Sigma_star_neg_half_log_sq_resid_vec);
//		System.out.println("V_beta");
//		V_beta.print(3, 5);
//		System.out.println("Beta_vec");
//		Beta_vec.print(3, 5);
		Matrix z = new Matrix(p + 1, 1);
		for (int i = 0; i < p + 1; i++){
			z.set(i, 0, StatToolbox.sample_from_std_norm_dist());
		}
//		System.out.println("z");
//		z.print(3, 5);	
		
		//return everything
		HashMap<String, Matrix> return_obj = new HashMap<String, Matrix>();
		return_obj.put("beta_vec", Beta_vec);
		return_obj.put("beta_draw", Beta_vec.plus(V_beta.times(z)));
		
		return return_obj;
	}

	private void SampleTausqForLMSigsqs(Matrix resids, int sample_num) {
		double sse = 0;
		for (int i = 0; i < n + p + 1; i++){
			sse += Math.pow(resids.get(i, 0), 2); 
		}
		System.out.println("sse: " + sse);
		gibbs_samples_of_tausq_for_lm_sigsqs[sample_num] = 1; //SSSSSSSSSSSSSSSSS
//			StatToolbox.sample_from_inv_gamma((hyper_nu_sigsq + n + p + 1) / 2, 2 / (sse + hyper_nu_sigsq * hyper_lambda_sigsq), this);
		
		System.out.println("tausq draw: " + gibbs_samples_of_tausq_for_lm_sigsqs[sample_num]);
	}
	
	private void SampleSigsqsViaLM(int sample_num) {
//		double sigsq_from_vanilla_bart = gibbs_samples_of_sigsq[sample_num];
		
		
		
//		double[] beta_lm_sigsq = gibbs_samples_of_betas_for_lm_sigsqs[sample_num];
		double[] beta_lm_sigsq = {0, 0.5};
//		double[] beta_lm_sigsq = {0, 2, 2};
		double tausq_lm_sigsq = gibbs_samples_of_tausq_for_lm_sigsqs[sample_num];
		double[] sigsqs_gibbs_sample = gibbs_samples_of_sigsq_hetero[sample_num]; //pointer to what we need to populate
//		double[] ln_sigsqs = new double[n];
		for (int i = 0; i < n; i++){
			//initialize to be the intercept
			double x_trans_beta_lm_sigsq = beta_lm_sigsq[0];
			for (int j = 1; j <= p; j++){
				x_trans_beta_lm_sigsq += X_y.get(i)[j - 1] * beta_lm_sigsq[j];
			}
//			sigsqs_gibbs_sample[i] = Math.exp(StatToolbox.sample_from_norm_dist(x_trans_beta_lm_sigsq, tausq_lm_sigsq));
//			sigsqs_gibbs_sample[i] = Math.exp(x_trans_beta_lm_sigsq); //SSSSSSSSSSSSSSSSSSSSSS
//			ln_sigsqs[i] = x_trans_beta_lm_sigsq;
//			sigsqs_gibbs_sample[i] = sigsq_from_vanilla_bart; 
			sigsqs_gibbs_sample[i] = Math.pow(transform_y(Math.sqrt(Math.exp(x_trans_beta_lm_sigsq))), 2); //SSSSSSSSSSSSSSSSSSSSSS
		}
//		System.out.println("ln_sigsq estimates: " + Tools.StringJoin(ln_sigsqs));
		System.out.println("sigsq estimates: " + Tools.StringJoin(sigsqs_gibbs_sample));
	}
	
	private void SampleMusF2(int sample_num, CGMBARTTreeNode node) {
		System.out.println("\n\nGibbs sample_num: " + sample_num + "  Mus \n" + "----------------------------------------------------");
		double[] current_sigsqs = gibbs_samples_of_sigsq_hetero[sample_num - 1];
		assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(node, current_sigsqs);
		sigsq_from_vanilla_bart = gibbs_samples_of_sigsq[sample_num - 1];
	}	
	
	protected void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(CGMBARTTreeNode node, double[] sigsqs) {
//		System.out.println("assignLeafValsUsingPosteriorMeanAndCurrentSigsq sigsqs: " + Tools.StringJoin(sigsqs));
		if (node.isLeaf){
			
//			System.out.println("sigsq_from_vanilla_bart: " + sigsq_from_vanilla_bart + " 1 / sigsq_from_vanilla_bart: " + 1 / sigsq_from_vanilla_bart);
			System.out.println("n = " + node.n_eta + " n over sigsq_from_vanilla_bart: " + node.n_eta / sigsq_from_vanilla_bart);
			
			//update ypred
			double posterior_var = calcLeafPosteriorVarF2(node, sigsqs);
			//draw from posterior distribution
			double posterior_mean = calcLeafPosteriorMeanF2(node, posterior_var, sigsqs);
//			System.out.println("assignLeafVals n_k = " + node.n_eta + " sum_nk_sq = " + Math.pow(node.n_eta, 2) + " node = " + node.stringLocation(true));
//			System.out.println("node responses: " + Tools.StringJoin(node.responses));
			node.y_pred = StatToolbox.sample_from_norm_dist(posterior_mean, posterior_var);
			
			double posterior_mean_untransformed = un_transform_y(posterior_mean);
			double posterior_sigma_untransformed = un_transform_y(Math.sqrt(posterior_var));
			double y_pred_untransformed = un_transform_y(node.y_pred);
			if (node.avg_response_untransformed() > 9){ 
				double posterior_mean_vanilla_un = un_transform_y(node.sumResponses() / sigsq_from_vanilla_bart / (1 / hyper_sigsq_mu + node.n_eta / sigsq_from_vanilla_bart));
				System.out.println("posterior_mean in BART = " + posterior_mean_vanilla_un);
				
				System.out.println("posterior_mean in HBART = " + posterior_mean_untransformed + 
						" node.avg_response = " + node.avg_response_untransformed() + 
						" y_pred_untransformed = " + y_pred_untransformed + 
						" posterior_sigma = " + posterior_sigma_untransformed + 
						" hyper_sigsq_mu = " + hyper_sigsq_mu);
			}
			
			
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
//			System.out.println("y_i = " + node.responses[ell] + " sigsq_i = " + sigsqs[node.indicies[ell]]);
			numerator += node.responses[ell] / sigsqs[node.indicies[ell]];
		}
//		System.out.println("calcLeafPosteriorMeanF2 numerator: " + numerator);
		return numerator * posterior_var;
	}

	private double calcLeafPosteriorVarF2(CGMBARTTreeNode node, double[] sigsqs) {
//		System.out.println("calcLeafPosteriorVarF2 sigsqs: " + Tools.StringJoin(sigsqs));
		double sum_one_over_sigsqs_leaf = 0;
//		System.out.print(" 1 / sigsqs: ");
		for (int index : node.indicies){
//			System.out.print( 1 / sigsqs[index] + ", ");
			sum_one_over_sigsqs_leaf += 1 / sigsqs[index];
		}
//		System.out.print("\n");
		System.out.println("sum_one_over_sigsqs_leaf: " + sum_one_over_sigsqs_leaf);
		return 1 / (1 / hyper_sigsq_mu + sum_one_over_sigsqs_leaf);
	}

	/**
	 * We run the default initialization plus all initializations for our sigsq model
	 */
	protected void InitGibbsSamplingData(){
		super.InitGibbsSamplingData();
		if (use_heteroskedasticity){
			gibbs_samples_of_sigsq_hetero = new double[num_gibbs_total_iterations + 1][n];	
			gibbs_samples_of_sigsq_hetero_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in][n];
			gibbs_samples_of_betas_for_lm_sigsqs = new double[num_gibbs_total_iterations + 1 ][p + 1];
			gibbs_samples_of_betas_for_lm_sigsqs_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in][p + 1];
			gibbs_samples_of_tausq_for_lm_sigsqs = new double[num_gibbs_total_iterations + 1];
			gibbs_samples_of_tausq_for_lm_sigsqs[0] = IntialTauSqLM;
			gibbs_samples_of_tausq_for_lm_sigsqs_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in];
		}	
	}	
	
	/**
	 * Instead of just setting one sigsq to the initial value, set sigsq's for all n observations to the initial value
	 */
	private void InitizializeSigsqF2() {
		double[] initial_sigsqs = gibbs_samples_of_sigsq_hetero[0];
		for (int i = 0; i < n; i++){
			initial_sigsqs[i] = INITIAL_SIGSQ;
		}	
	}	
	
	/////////////nothing but scaffold code below, do not alter!

	protected void InitizializeSigsq() {
		super.InitizializeSigsq();
		if (use_heteroskedasticity){
			InitizializeSigsqF2();
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
	
	protected double calcLnLikRatioChange(CGMBARTTreeNode eta, CGMBARTTreeNode eta_star) {
		if (use_heteroskedasticity){
			return calcLnLikRatioChangeF2(eta, eta_star);
		}
		return super.calcLnLikRatioChange(eta, eta_star);
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

	public void setHyperBetaSigsq(double hyper_beta_sigsq){
		this.hyper_beta_sigsq = hyper_beta_sigsq;
	}

	/**
	 * The user specifies this flag. Once set, the functions in this class are used over the default homoskedastic functions
	 * in parent classes
	 */
	public void useHeteroskedasticity(){
		use_heteroskedasticity = true;
	}
}
