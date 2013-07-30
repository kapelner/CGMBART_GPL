package CGM_BART;

import java.util.ArrayList;
import Jama.Matrix;


public class CGMBART_F2_linear_heteroskedasticity extends CGMBART_F1_prior_cov_spec {
	private static final long serialVersionUID = -3069428133597923502L;


//	private static final boolean GAMERMAN = true;

	protected boolean use_linear_heteroskedasticity_model;
	
	protected Matrix hyper_gamma_mean_vec;
	protected Matrix hyper_gamma_var_mat;
	
	/** the variance of the errors as well as other things necessary for Gibbs sampling */
	protected double[][] gibbs_samples_of_sigsq_hetero;
	protected double[][] gibbs_samples_of_sigsq_hetero_after_burn_in;	
	protected Matrix[] gibbs_samples_of_gamma_for_lm_sigsqs;
	protected Matrix[] gibbs_samples_of_gamma_for_lm_sigsqs_after_burn_in;
	
	protected int[] m_h_num_accept_over_gibbs_samples;
	protected int[] m_h_num_accept_over_gibbs_samples_after_burn_in;

	/** convenience caches */
	private Matrix Xmat_with_intercept;
	private ArrayList<Matrix> x_is;
	private Matrix Bmat;
	private Matrix[] mu_vec_B_terms_j;
	private double[] tausq_j;
	private int[][] minus_j_indices;
	private Matrix Sigmainv_times_hyper_gamma_mean_vec;
	private Matrix Xmat_with_intercept_transpose;
	private Matrix halves_times_Xmat_with_intercept_transpose;


	private Matrix XtXinvXt;


	public void Build(){
		super.Build();
		if (use_linear_heteroskedasticity_model){
			for (int j = 0; j < p + 1; j++){
				double prop_accepted_tot = m_h_num_accept_over_gibbs_samples[j] / (double) num_gibbs_total_iterations;
				System.out.println("prop gibbs accepted tot for j = " + j + ": " + prop_accepted_tot);
			}
			System.out.println("\n\n");
			for (int j = 0; j < p + 1; j++){
				double prop_accepted_after_burn_in = m_h_num_accept_over_gibbs_samples_after_burn_in[j] / (double) (num_gibbs_total_iterations - num_gibbs_burn_in);
				System.out.println("prop gibbs accepted after burn in for j = " + j + ": " + prop_accepted_after_burn_in);
			}
			System.out.println("\n\n");
			
			
			
			double gamma_j_avg = 0;
			double gamma_j_sd = 0;
			for (int j = 0; j < p + 1; j++){
				double[] gibbs_samples_gamma_j = new double[num_gibbs_total_iterations - num_gibbs_burn_in];
				for (int g = num_gibbs_burn_in; g < num_gibbs_total_iterations; g++){
					gibbs_samples_gamma_j[g - num_gibbs_burn_in] = gibbs_samples_of_gamma_for_lm_sigsqs[g].get(j, 0);
					gamma_j_avg = StatToolbox.sample_average(gibbs_samples_gamma_j);
					gamma_j_sd = StatToolbox.sample_standard_deviation(gibbs_samples_gamma_j);
				}
				System.out.println("gamma_" + j + " = " + Tools.two_digit_format.format(gamma_j_avg) + " +- " + Tools.two_digit_format.format(gamma_j_sd) +
						((gamma_j_avg - 2 * gamma_j_sd < 0 && gamma_j_avg + 2 * gamma_j_sd > 0) ? " => plausibly 0" : " => *NOT* plausibly 0"));
			}
		}
	}
	
	public double[][] getGammas(){
		double[][] gammas = new double[num_gibbs_total_iterations][p + 1];
		for (int g = 0; g < num_gibbs_total_iterations; g++){
			for (int j = 0; j < p + 1; j++){
				gammas[g][j] = gibbs_samples_of_gamma_for_lm_sigsqs[g].get(j, 0); 
			}
			
		}
		return gammas;
	}

	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		if (use_linear_heteroskedasticity_model){
			System.out.println("use_linear_heteroskedasticity_model   n: " + n + " p: " + p);
						
			//precompute X as a Matrix object
			Xmat_with_intercept = new Matrix(n, p + 1);
			//the top part is just the original X training matrix
			for (int i = 0; i < n; i++){
				for (int j = 0; j < p + 1; j++){
					if (j == 0){
						Xmat_with_intercept.set(i, j, 1); //this is the intercept
					}
					else {
						Xmat_with_intercept.set(i, j, X_y.get(i)[j - 1]);
					}
				}
			}
			//the bottom portion is the identity matrix
//			for (int i = n; i < n + p + 1; i++){
//				Xmat_star.set(i, i - n, 1);
//			}
			
			Xmat_with_intercept_transpose = Xmat_with_intercept.transpose();
			
			XtXinvXt = (Xmat_with_intercept_transpose.times(Xmat_with_intercept)).inverse().times(Xmat_with_intercept_transpose);
			
//			System.out.println("Xmat_star");
//			Xmat_with_intercept.print(3, 5);
			
			
		
			//now we make another convenience structure
			
			x_is = new ArrayList<Matrix>(n);
			for (int i = 0; i < n; i++){
				Matrix x_i = new Matrix(1, p + 1);
				for (int j = 0; j < p + 1; j++){
					x_i.set(0, j, Xmat_with_intercept.get(i, j));
				}
				x_is.add(x_i); //this goes in order, so it's okay
			}
		
			//set hyperparameters
			hyper_gamma_mean_vec = new Matrix(p + 1, 1);

			
			
			
			hyper_gamma_var_mat = new Matrix(p + 1, p + 1);
			for (int j = 0; j < p + 1; j++){
				hyper_gamma_var_mat.set(j, j, 0.001);
			}
			
			///////////informed model
			hyper_gamma_mean_vec.set(0, 0, 1);
			hyper_gamma_mean_vec.set(1, 0, 0.3);
			hyper_gamma_mean_vec.set(2, 0, 0.5);
			hyper_gamma_mean_vec.set(3, 0, 0.3);
			hyper_gamma_mean_vec.set(4, 0, 0.2);
			hyper_gamma_mean_vec.set(5, 0, 0.5);
			
			
			Matrix halves = new Matrix(p + 1, p + 1);
			for (int j = 0; j < p + 1; j++){
				halves.set(j, j, 0.5);
			}
			
//			System.out.println("hyper_gamma_var_mat");
//			hyper_gamma_var_mat.print(3, 5);
			
			halves_times_Xmat_with_intercept_transpose = halves.times(Xmat_with_intercept_transpose);
			
			//now we can cache intermediate values we'll use everywhere
			Matrix Sigmainv = hyper_gamma_var_mat.inverse();
			
//			System.out.println("Sigmainv");
//			Sigmainv.print(3, 5);
			
			Sigmainv_times_hyper_gamma_mean_vec = Sigmainv.times(hyper_gamma_mean_vec);
			Bmat = Sigmainv.plus(halves_times_Xmat_with_intercept_transpose.times(Xmat_with_intercept)).inverse();
			
//			System.out.println("Bmat");
//			Bmat.print(3, 5);
			
			mu_vec_B_terms_j = new Matrix[p + 1];
			tausq_j = new double[p + 1];
			minus_j_indices = new int[p + 1][p];
			
			for (int j = 0; j < p + 1; j++){
//				System.out.println("---------------------------- j = " + j);
				
				for (int k = 0; k < p; k++){
					minus_j_indices[j][k] = k < j ? k : (k + 1);
				}
				
				
				Matrix Bmat_j_j = Bmat.getMatrix(j, j, j, j);
				Matrix Bmat_j_minus_j = Bmat.getMatrix(j, j, minus_j_indices[j]);
				Matrix Bmat_minus_j_j = Bmat.getMatrix(minus_j_indices[j], j, j);
				Matrix Bmat_minus_j_minus_j = Bmat.getMatrix(minus_j_indices[j], minus_j_indices[j]);

				mu_vec_B_terms_j[j] = Bmat_j_minus_j.times(Bmat_minus_j_minus_j.inverse());
				tausq_j[j] = Bmat_j_j.minus(mu_vec_B_terms_j[j].times(Bmat_minus_j_j)).get(0, 0);

//				System.out.println("minus_j = " + Tools.StringJoin(minus_j_indices[j]));
//				System.out.println("Bmat_j_j");
//				Bmat_j_j.print(3, 5);
//				
//				System.out.println("Bmat_j_minus_j");
//				Bmat_j_minus_j.print(3, 5);
//				
//				System.out.println("Bmat_minus_j_j");
//				Bmat_minus_j_j.print(3, 5);
//				
//				System.out.println("Bmat_minus_j_minus_j");
//				Bmat_minus_j_minus_j.print(3, 5);
			}
			
		}
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
		double[] sigsqs = gibbs_samples_of_sigsq_hetero[gibbs_sample_num - 1];
				
		double sum_inv_sigsq_ell_star = 0;
		double sum_responses_weighted_by_inv_sigsq_ell_star = 0;
		for (int i = 0; i < eta_star.left.n_eta; i++){
			int index = eta_star.left.indicies[i];
			double response_i = eta_star.left.responses[i];
			double sigsq_i = sigsqs[index];
			sum_inv_sigsq_ell_star += 1 / sigsq_i;
			sum_responses_weighted_by_inv_sigsq_ell_star += response_i / sigsq_i;
		}
		
		double sum_inv_sigsq_r_star = 0;
		double sum_responses_weighted_by_inv_sigsq_r_star = 0;
		for (int i = 0; i < eta_star.right.n_eta; i++){
			int index = eta_star.right.indicies[i];
			double response_i = eta_star.right.responses[i];
			double sigsq_i = sigsqs[index];
			sum_inv_sigsq_r_star += 1 / sigsq_i;
			sum_responses_weighted_by_inv_sigsq_r_star += response_i / sigsq_i;
		}
		
		double sum_inv_sigsq_ell = 0;
		double sum_responses_weighted_by_inv_sigsq_ell = 0;
		for (int i = 0; i < eta.left.n_eta; i++){
			int index = eta.left.indicies[i];
			double response_i = eta.left.responses[i];
			double sigsq_i = sigsqs[index];
			sum_inv_sigsq_ell += 1 / sigsq_i;
			sum_responses_weighted_by_inv_sigsq_ell += response_i / sigsq_i;
		}
		
		double sum_inv_sigsq_r = 0;
		double sum_responses_weighted_by_inv_sigsq_r = 0;
		for (int i = 0; i < eta.right.n_eta; i++){
			int index = eta.right.indicies[i];
			double response_i = eta.right.responses[i];
			double sigsq_i = sigsqs[index];
			sum_inv_sigsq_r += 1 / sigsq_i;
			sum_responses_weighted_by_inv_sigsq_r += response_i / sigsq_i;
		}	
		
		double one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_ell_star = 1 + hyper_sigsq_mu * sum_inv_sigsq_ell_star;
		double one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_r_star = 1 + hyper_sigsq_mu * sum_inv_sigsq_r_star;
		double one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_ell = 1 + hyper_sigsq_mu * sum_inv_sigsq_ell;
		double one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_r = 1 + hyper_sigsq_mu * sum_inv_sigsq_r;
		
		double a = Math.log(one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_ell_star);
		double b = Math.log(one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_r_star);
		double c = Math.log(one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_ell);
		double d = Math.log(one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_r);
		
		double e = Math.pow(sum_responses_weighted_by_inv_sigsq_ell_star, 2) / one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_ell_star;
		double f = Math.pow(sum_responses_weighted_by_inv_sigsq_r_star, 2) / one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_r_star;
		double g = Math.pow(sum_responses_weighted_by_inv_sigsq_ell, 2) / one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_ell;
		double h = Math.pow(sum_responses_weighted_by_inv_sigsq_r, 2) / one_plus_sigsq_mu_times_sum_one_over_sigsq_i_n_r;		
		
		return 0.5 * (a + b - c - d) + hyper_sigsq_mu / 2 * (e + f - g - h);
	}		
	
	private void SampleSigsqF2(int sample_num, double[] es) {
//		System.out.println("\n\nGibbs sample_num: " + sample_num + "  Sigsqs \n" + "----------------------------------------------------");
//		System.out.println("es: " + Tools.StringJoin(es));

//		System.out.println("s^2_e = " + StatToolbox.sample_variance(es));
		
		double[] es_sq = new double[n];
		for (int i = 0; i < n; i++){
			es_sq[i] = un_transform_sigsq(Math.pow(es[i], 2));
		}
//		System.out.println("es_sq: " + Tools.StringJoin(es_sq));
		
		//now we need to draw a gamma
		Matrix gamma_draw = DrawGammaVecViaMH(gibbs_samples_of_gamma_for_lm_sigsqs[sample_num - 1], es_sq, sample_num);
		gibbs_samples_of_gamma_for_lm_sigsqs[sample_num] = gamma_draw;

		//now set the sigsqs just by doing exp(x_i^T \gammavec)
//		double[] new_sigsqs = new double[n];
		for (int i = 0; i < n; i++){
			double sigsq_i = Math.exp(x_is.get(i).times(gamma_draw).get(0, 0));
//			new_sigsqs[i] = sigsq_i;
			gibbs_samples_of_sigsq_hetero[sample_num][i] = transform_sigsq(sigsq_i); //make sure we re-transform them
		}	
//		System.out.println("new_sigsqs: " + Tools.StringJoin(new_sigsqs));
	}

	private Matrix DrawGammaVecViaMH(Matrix gamma, double[] es_sq, int sample_num) {
		
		Matrix ln_es_sq = new Matrix(n, 1);
		for (int i = 0; i < n; i++){
			ln_es_sq.set(i, 0, Math.log(es_sq[i]));
		}
		
		Matrix gamma_star = XtXinvXt.times(ln_es_sq);
		
		System.out.println("gamma");
		gamma.print(3, 5);
		
		System.out.println("gamma_star");
		gamma_star.print(3, 5);
		
		return gamma_star;
		
		/**
		System.out.println("\nDrawGammaVecViaMH g = " + sample_num + "\n");
		
		Matrix gamma_copy = (Matrix) gamma.clone();
		//do the whole thing for each dimension separately
		for (int j = 0; j < p + 1; j++){
			
			System.out.println("\n   Sampling j = " + j + " g = " + sample_num + "\n");
			
			double gamma_j = gamma_copy.get(j, 0);
			
		
			Matrix gamma_star = (Matrix) gamma_copy.clone();
			
			System.out.println("gamma_copy");
			gamma_copy.print(3, 5);
			
			//get the d vectors
			double sum_x_i_times_gamma = 0;
			double sum_es_sq_over_exp_x_i_times_gamma = 0;
//			double[] exp_x_i_times_gammas = new double[n];
//			double[] es_sq_over_exp_x_i_times_gammas = new double[n];
			Matrix d = new Matrix(n, 1);
			for (int i = 0; i < n; i++){
				double x_i_times_gamma = x_is.get(i).times(gamma_copy).get(0, 0);
				double exp_x_i_times_gamma = Math.exp(x_i_times_gamma);				
//				exp_x_i_times_gammas[i] = exp_x_i_times_gamma;
				d.set(i, 0, x_i_times_gamma + es_sq[i] / exp_x_i_times_gamma - 1);
				//cache for later
				sum_x_i_times_gamma += x_i_times_gamma;
				sum_es_sq_over_exp_x_i_times_gamma += es_sq[i] / exp_x_i_times_gamma;
//				es_sq_over_exp_x_i_times_gammas[i] = es_sq[i] / exp_x_i_times_gamma;
			}
//			System.out.println("exp_x_i_times_gammas: " + Tools.StringJoin(exp_x_i_times_gammas));
//			System.out.println("es_sq_over_exp_x_i_times_gammas: " + Tools.StringJoin(es_sq_over_exp_x_i_times_gammas));
			
						
			
			Matrix a_gamma = Bmat.times(Sigmainv_times_hyper_gamma_mean_vec.plus(halves_times_Xmat_with_intercept_transpose.times(d)));
			
			double a_gamma_j = a_gamma.get(j, 0);
			Matrix a_gamma_minus_j = a_gamma.getMatrix(minus_j_indices[j], 0, 0);
			Matrix gamma_minus_j = gamma_copy.getMatrix(minus_j_indices[j], 0, 0);
			
			
			double mu_j = a_gamma_j + mu_vec_B_terms_j[j].times(gamma_minus_j.minus(a_gamma_minus_j)).get(0, 0);
					
			//draw gamma^*_j and shove it in the correct place inside gamma_star_copy
			double gamma_star_j = StatToolbox.sample_from_norm_dist(mu_j, tausq_j[j]);
			gamma_star.set(j, 0, gamma_star_j);
			
			System.out.println("gamma_star");
			gamma_star.print(3, 5);
			
			

			double sum_x_i_times_gamma_star = 0;
			double sum_es_sq_over_exp_x_i_times_gamma_star = 0;
			Matrix d_star = new Matrix(n, 1);
			for (int i = 0; i < n; i++){
				double x_i_times_gamma_star = x_is.get(i).times(gamma_star).get(0, 0);
				double exp_x_i_times_gamma_star = Math.exp(x_i_times_gamma_star);
				d_star.set(i, 0, x_i_times_gamma_star + es_sq[i] / exp_x_i_times_gamma_star - 1);
				//cache for later
				sum_x_i_times_gamma_star += x_i_times_gamma_star;
				sum_es_sq_over_exp_x_i_times_gamma_star += es_sq[i] / exp_x_i_times_gamma_star;
			}

			

			
			Matrix a_gamma_star = Bmat.times(Sigmainv_times_hyper_gamma_mean_vec.plus(halves_times_Xmat_with_intercept_transpose.times(d_star)));			
		
			double a_gamma_star_j = a_gamma_star.get(j, 0);
			Matrix a_gamma_star_minus_j = a_gamma_star.getMatrix(minus_j_indices[j], 0, 0);
			Matrix gamma_star_minus_j = gamma_star.getMatrix(minus_j_indices[j], 0, 0);
			
			
			//note gamma_minus_j is the same as gamma_star_minus_j
			double mu_j_star = a_gamma_star_j + mu_vec_B_terms_j[j].times(gamma_star_minus_j.minus(a_gamma_star_minus_j)).get(0, 0);
			
			
			double log_prop_prob_gamma_star_j_to_gamma_j = 1 / (tausq_j[j]) * Math.pow(gamma_j - mu_j_star, 2);
			double log_prop_prob_gamma_j_to_gamma_star_j = 1 / (tausq_j[j]) * Math.pow(gamma_star_j - mu_j, 2);
			
			//last term in these log probs
			double gamma_minus_hyper_gamma_sq_over_hyper_var = Math.pow(gamma_j - hyper_gamma_mean_vec.get(j, 0), 2) / hyper_gamma_var_mat.get(j, j);
			double gamma_star_minus_hyper_gamma_sq_over_hyper_var = Math.pow(gamma_star_j - hyper_gamma_mean_vec.get(j, 0), 2) / hyper_gamma_var_mat.get(j, j);
			
				
			
			double log_prop_prob_gamma_star_j = sum_x_i_times_gamma_star + sum_es_sq_over_exp_x_i_times_gamma_star + gamma_star_minus_hyper_gamma_sq_over_hyper_var;
			System.out.println("*--  sum_x_i_times_gamma_star: " + sum_x_i_times_gamma_star);
			System.out.println("*--  sum_es_sq_over_exp_x_i_times_gamma_star: " + sum_es_sq_over_exp_x_i_times_gamma_star);
			System.out.println("*--  gamma_star_minus_hyper_gamma_sq_over_hyper_var: " + gamma_star_minus_hyper_gamma_sq_over_hyper_var);
			
			
			double log_prop_prob_gamma_j = sum_x_i_times_gamma + sum_es_sq_over_exp_x_i_times_gamma + gamma_minus_hyper_gamma_sq_over_hyper_var;
			System.out.println("--  sum_x_i_times_gamma: " + sum_x_i_times_gamma);
			System.out.println("--  sum_es_sq_over_exp_x_i_times_gamma: " + sum_es_sq_over_exp_x_i_times_gamma);
			System.out.println("--  gamma_minus_hyper_gamma_sq_over_hyper_var: " + gamma_minus_hyper_gamma_sq_over_hyper_var);
			
			
			
			double mh_ratio = -0.5 * (log_prop_prob_gamma_star_j_to_gamma_j - 
						log_prop_prob_gamma_j_to_gamma_star_j + 
						log_prop_prob_gamma_star_j - 
						log_prop_prob_gamma_j);
//			double mh_ratio = log_prop_prob_gamma_star - log_prop_prob_gamma;
			
//			System.out.println("\n\n log_prop_prob_gamma_star: " + log_prop_prob_gamma_star + " - log_prop_prob_gamma: " + log_prop_prob_gamma);
//			System.out.println("log_prop_prob_gamma_star_j_to_gamma_j: " + log_prop_prob_gamma_star_j_to_gamma_j + 
//					" - log_prop_prob_gamma_j_to_gamma_star_j: " + log_prop_prob_gamma_j_to_gamma_star_j + 
//					" + log_prop_prob_gamma_star_j: " + log_prop_prob_gamma_star_j + 
//					" - log_prop_prob_gamma_j: " + log_prop_prob_gamma_j);
			
			System.out.println("diff jumping: " + (log_prop_prob_gamma_star_j_to_gamma_j - log_prop_prob_gamma_j_to_gamma_star_j));
			System.out.println("diff lik: " + (log_prop_prob_gamma_star_j - log_prop_prob_gamma_j));
			
			double log_r = Math.log(StatToolbox.rand());
			
			System.out.println("log_r = " + log_r + " mh_ratio = " + mh_ratio);
			
			if (log_r < mh_ratio){
				System.out.println("VAR ACCEPT MH for j = " + j + " g = " + sample_num);
				m_h_num_accept_over_gibbs_samples[j]++;
				if (sample_num > num_gibbs_burn_in){
					m_h_num_accept_over_gibbs_samples_after_burn_in[j]++;
				}
				gamma_copy = gamma_star;
			} 
			else {
				System.out.println("VAR REJECT MH for j = " + j + " g = " + sample_num);
			}
			
//			if (sample_num > 20){
//				try {
//					Thread.sleep(1000);
//				} catch (InterruptedException e) {
//					e.printStackTrace();
//				}
//			}
		}
		
		//this is all dimensions done - some of the p+1 will be changed, some will not be changed
		return gamma_copy;
		
		*/
		
		
		
		
		
		//-----------------------------------------------------
		/**
		
		
		
		
		
		
		
		
		
		//this is the M-H step
		
		double[] exp_x_i_times_gammas = new double[n];
		double[] es_sq_over_exp_x_i_times_gammas = new double[n];
		double sum_x_i_times_gamma = 0;
		double sum_es_sq_over_exp_x_i_times_gamma = 0;
		
		Matrix d = new Matrix(n, 1);
		for (int i = 0; i < n; i++){
			double x_i_times_gamma = x_is.get(i).times(gamma).get(0, 0);
			double exp_x_i_times_gamma = Math.exp(x_i_times_gamma);
			exp_x_i_times_gammas[i] = exp_x_i_times_gamma;
			d.set(i, 0, x_i_times_gamma + es_sq[i] / exp_x_i_times_gamma - 1);
			//cache for later
			sum_x_i_times_gamma += x_i_times_gamma;
			sum_es_sq_over_exp_x_i_times_gamma += es_sq[i] / exp_x_i_times_gamma;
			es_sq_over_exp_x_i_times_gammas[i] = es_sq[i] / exp_x_i_times_gamma;
		}
//		System.out.println("exp_x_i_times_gammas: " + Tools.StringJoin(exp_x_i_times_gammas));
//		System.out.println("es_sq_over_exp_x_i_times_gammas: " + Tools.StringJoin(es_sq_over_exp_x_i_times_gammas));
//		System.out.println("dims Bmatinv: " + Bmatinv.getRowDimension() + " x " + Bmatinv.getColumnDimension());
//		System.out.println("dims Sigmainv_times_hyper_gamma_mean_vec: " + Sigmainv_times_hyper_gamma_mean_vec.getRowDimension() + " x " + Sigmainv_times_hyper_gamma_mean_vec.getColumnDimension());
//		System.out.println("dims halves_times_Xmat_with_intercept_transpose: " + halves_times_Xmat_with_intercept_transpose.getRowDimension() + " x " + halves_times_Xmat_with_intercept_transpose.getColumnDimension());
//		System.out.println("dims d: " + d.getRowDimension() + " x " + d.getColumnDimension());
		
		
		
//		System.out.println("Sigmainv_times_hyper_gamma_mean_vec");
//		Sigmainv_times_hyper_gamma_mean_vec.print(3, 5);
//		System.out.println("d");
//		d.print(3, 5);
		
		
		
		Matrix gamma_star = null;
		Matrix a = null;
		
		if (!GAMERMAN){
			gamma_star = StatToolbox.sample_from_mult_norm_dist(gamma, Imat_p);
		}
		else {
			
			gamma_star = StatToolbox.sample_from_mult_norm_dist_with_Sigma_sqrt(a, Bmat_sqrt);
		}
		

		System.out.println("gamma");
		gamma.print(3, 5);
		
		System.out.println("gamma_star");
		gamma_star.print(3, 5);

		
		double sum_x_i_times_gamma_star = 0;
		double sum_es_sq_over_exp_x_i_times_gamma_star = 0;
		Matrix d_star = new Matrix(n, 1);
		for (int i = 0; i < n; i++){
			double x_i_times_gamma_star = x_is.get(i).times(gamma_star).get(0, 0);
			double exp_x_i_times_gamma_star = Math.exp(x_i_times_gamma_star);
			d_star.set(i, 0, x_i_times_gamma_star + es_sq[i] / exp_x_i_times_gamma_star - 1);
			//cache for later
			sum_x_i_times_gamma_star += x_i_times_gamma_star;
			sum_es_sq_over_exp_x_i_times_gamma_star += es_sq[i] / exp_x_i_times_gamma_star;
		}
		double sum_gamma_min_hyper_gamma_sq_over_hyper_var = 0;
		double sum_gamma_star_min_hyper_gamma_star_sq_over_hyper_var = 0;
		
		for (int j = 0; j < p + 1; j++){
			sum_gamma_min_hyper_gamma_sq_over_hyper_var += Math.pow(gamma.get(j, 0) - hyper_gamma_mean_vec.get(j, 0), 2) / hyper_gamma_var_mat.get(j, j);
			sum_gamma_star_min_hyper_gamma_star_sq_over_hyper_var += Math.pow(gamma_star.get(j, 0) - hyper_gamma_mean_vec.get(j, 0), 2) / hyper_gamma_var_mat.get(j, j);
		}
		

		System.out.println("sum_x_i_times_gamma_star: " + sum_x_i_times_gamma_star + " sum_es_sq_over_exp_x_i_times_gamma_star: " + sum_es_sq_over_exp_x_i_times_gamma_star + " sum_gamma_star_min_hyper_gamma_star_sq_over_hyper_var: " + sum_gamma_star_min_hyper_gamma_star_sq_over_hyper_var);
		double log_prop_prob_gamma_star = -0.5 * (sum_x_i_times_gamma_star + sum_es_sq_over_exp_x_i_times_gamma_star + sum_gamma_star_min_hyper_gamma_star_sq_over_hyper_var);
		System.out.println("sum_x_i_times_gamma: " + sum_x_i_times_gamma + " sum_es_sq_over_exp_x_i_times_gamma: " + sum_es_sq_over_exp_x_i_times_gamma + " sum_gamma_min_hyper_gamma_sq_over_hyper_var: " + sum_gamma_min_hyper_gamma_sq_over_hyper_var);
		double log_prop_prob_gamma = -0.5 * (sum_x_i_times_gamma + sum_es_sq_over_exp_x_i_times_gamma + sum_gamma_min_hyper_gamma_sq_over_hyper_var);
		
		
						
		
		
		double mh_ratio = 0;
		

		

		if (!GAMERMAN){
			
			
			mh_ratio = log_prop_prob_gamma_star - log_prop_prob_gamma;
			
			System.out.println("\n\n log_prop_prob_gamma_star: " + log_prop_prob_gamma_star + " - log_prop_prob_gamma: " + log_prop_prob_gamma);
			
		}
		else {
			
			Matrix a_star = Bmat.times(Sigmainv_times_hyper_gamma_mean_vec.plus(halves_times_Xmat_with_intercept_transpose.times(d_star)));			
//			
			Matrix gamma_minus_a_star = gamma.minus(a_star);
			Matrix gamma_star_minus_a = gamma_star.minus(a);
//			
//			System.out.println("Bmat");
//			Bmat.print(3, 5);
//			System.out.println("halves_times_Xmat_with_intercept_transpose.times(d)");
//			halves_times_Xmat_with_intercept_transpose.times(d).print(3, 5);
//			System.out.println("a");
//			a.print(3, 5);
			
			
//			System.out.println("d_star");
//			d_star.print(3, 5);
//			System.out.println("a_star");
//			a_star.print(3, 5);
//			System.out.println("gamma_minus_a_star");
//			gamma_minus_a_star.print(3, 5);
//			System.out.println("gamma_star_minus_a");
//			gamma_star_minus_a.print(3, 5);


			double log_prop_prob_gamma_star_to_gamma = -0.5 * (gamma_minus_a_star.transpose()).times(Bmatinv).times(gamma_minus_a_star).get(0, 0);
			double log_prop_prob_gamma_to_gamma_star = -0.5 * (gamma_star_minus_a.transpose()).times(Bmatinv).times(gamma_star_minus_a).get(0, 0);
			 
			mh_ratio = log_prop_prob_gamma_star_to_gamma - log_prop_prob_gamma_to_gamma_star + log_prop_prob_gamma_star - log_prop_prob_gamma;
//			double mh_ratio = log_prop_prob_gamma_star - log_prop_prob_gamma;
			
//			System.out.println("\n\n log_prop_prob_gamma_star: " + log_prop_prob_gamma_star + " - log_prop_prob_gamma: " + log_prop_prob_gamma);
			System.out.println("log_prop_prob_gamma_star_to_gamma: " + log_prop_prob_gamma_star_to_gamma + " - log_prop_prob_gamma_to_gamma_star: " + log_prop_prob_gamma_to_gamma_star + " + log_prop_prob_gamma_star: " + log_prop_prob_gamma_star + " - log_prop_prob_gamma: " + log_prop_prob_gamma);
			
		}
		
		System.out.println("mh_ratio: " + mh_ratio + " log_r: " + log_r);
		
		if (log_r < mh_ratio){
			System.out.println("VAR ACCEPT MH");
			m_h_num_accept_over_gibbs_samples++;
			if (sample_num > num_gibbs_burn_in){
				m_h_num_accept_over_gibbs_samples_after_burn_in++;
			}
			return gamma_star;
		}
		System.out.println("VAR REJECT MH");
		return gamma;
		*/
	} 

	private void SampleMusF2(int sample_num, CGMBARTTreeNode node) {
//		System.out.println("\n\nGibbs sample_num: " + sample_num + "  Mus \n" + "----------------------------------------------------");
		double[] current_sigsqs = gibbs_samples_of_sigsq_hetero[sample_num - 1];
		assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(node, current_sigsqs);
//		sigsq_from_vanilla_bart = gibbs_samples_of_sigsq[sample_num - 1];
	}	
	
	protected void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqsAndUpdateYhatsF2(CGMBARTTreeNode node, double[] sigsqs) {
//		System.out.println("assignLeafValsUsingPosteriorMeanAndCurrentSigsq sigsqs: " + Tools.StringJoin(sigsqs));
		if (node.isLeaf){
			
//			System.out.println("sigsq_from_vanilla_bart: " + sigsq_from_vanilla_bart + " 1 / sigsq_from_vanilla_bart: " + 1 / sigsq_from_vanilla_bart);
//			System.out.println("n = " + node.n_eta + " n over sigsq_from_vanilla_bart: " + node.n_eta / sigsq_from_vanilla_bart);
			
			//update ypred
			double posterior_var = calcLeafPosteriorVarF2(node, sigsqs);
			//draw from posterior distribution
			double posterior_mean = calcLeafPosteriorMeanF2(node, posterior_var, sigsqs);
//			System.out.println("assignLeafVals n_k = " + node.n_eta + " sum_nk_sq = " + Math.pow(node.n_eta, 2) + " node = " + node.stringLocation(true));
//			System.out.println("node responses: " + Tools.StringJoin(node.responses));
			node.y_pred = StatToolbox.sample_from_norm_dist(posterior_mean, posterior_var);
			
//			double posterior_mean_untransformed = un_transform_y(posterior_mean);
//			double posterior_sigma_untransformed = un_transform_y(Math.sqrt(posterior_var));
//			double y_pred_untransformed = un_transform_y(node.y_pred);
//			if (node.avg_response_untransformed() > 9){ 
//				double posterior_mean_vanilla_un = un_transform_y(node.sumResponses() / sigsq_from_vanilla_bart / (1 / hyper_sigsq_mu + node.n_eta / sigsq_from_vanilla_bart));
//				System.out.println("posterior_mean in BART = " + posterior_mean_vanilla_un);
				
//				System.out.println("posterior_mean in HBART = " + posterior_mean_untransformed + 
//						" node.avg_response = " + node.avg_response_untransformed() + 
//						" y_pred_untransformed = " + y_pred_untransformed + 
//						" posterior_sigma = " + posterior_sigma_untransformed + 
//						" hyper_sigsq_mu = " + hyper_sigsq_mu);
//			}
			
			
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
//		System.out.println("sum_one_over_sigsqs_leaf: " + sum_one_over_sigsqs_leaf);
		return 1 / (1 / hyper_sigsq_mu + sum_one_over_sigsqs_leaf);
	}

	/**
	 * We run the default initialization plus all initializations for our sigsq model
	 */
	protected void InitGibbsSamplingData(){
		super.InitGibbsSamplingData();
		if (use_linear_heteroskedasticity_model){
			gibbs_samples_of_sigsq_hetero = new double[num_gibbs_total_iterations + 1][n];	
			gibbs_samples_of_sigsq_hetero_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in][n];
			gibbs_samples_of_gamma_for_lm_sigsqs = new Matrix[num_gibbs_total_iterations + 1];
			gibbs_samples_of_gamma_for_lm_sigsqs[0] = new Matrix(p + 1, 1); //start it up
			
			m_h_num_accept_over_gibbs_samples = new int[p + 1];
			m_h_num_accept_over_gibbs_samples_after_burn_in = new int[p + 1];
			
			//set the beginning of the Gibbs chain to be the prior
			for (int j = 0; j < p + 1; j++){
				gibbs_samples_of_gamma_for_lm_sigsqs[0].set(j, 0, hyper_gamma_mean_vec.get(j, 0));
			}	
			
			gibbs_samples_of_gamma_for_lm_sigsqs_after_burn_in = new Matrix[num_gibbs_total_iterations - num_gibbs_burn_in];
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
		if (use_linear_heteroskedasticity_model){
			InitizializeSigsqF2();
		}
	}

	protected void SampleMus(int sample_num, CGMBARTTreeNode tree) {
		if (use_linear_heteroskedasticity_model){
			SampleMusF2(sample_num, tree);
		}
		else {
			super.SampleMus(sample_num, tree);
		}
	}	

	protected void SampleSigsq(int sample_num, double[] es) {
		if (use_linear_heteroskedasticity_model){
			SampleSigsqF2(sample_num, es);
		}
		else {
			super.SampleSigsq(sample_num, es);
		}		
	}

	protected double calcLnLikRatioGrow(CGMBARTTreeNode grow_node) {
		if (use_linear_heteroskedasticity_model){
			return calcLnLikRatioGrowF2(grow_node);
		}
		return super.calcLnLikRatioGrow(grow_node);
	}
	
	protected double calcLnLikRatioChange(CGMBARTTreeNode eta, CGMBARTTreeNode eta_star) {
		if (use_linear_heteroskedasticity_model){
			return calcLnLikRatioChangeF2(eta, eta_star);
		}
		return super.calcLnLikRatioChange(eta, eta_star);
	}

	/**
	 * The user specifies this flag. Once set, the functions in this class are used over the default homoskedastic functions
	 * in parent classes
	 */
	public void useLinearHeteroskedasticityModel(){
		use_linear_heteroskedasticity_model = true;
	}
	
	public void setHyper_gamma_mean_vec(double[] hyper_gamma_mean_vec) {
		for (int i = 0; i < hyper_gamma_mean_vec.length; i++){
			this.hyper_gamma_mean_vec.set(0, i, hyper_gamma_mean_vec[i]);
		}
	}

	public void setHyper_gamma_var_mat(double[] hyper_gamma_var_mat_diag) {
		for (int j = 0; j < hyper_gamma_var_mat_diag.length; j++){
			hyper_gamma_var_mat.set(j, j, hyper_gamma_var_mat_diag[j]);
		}
	}	
}
