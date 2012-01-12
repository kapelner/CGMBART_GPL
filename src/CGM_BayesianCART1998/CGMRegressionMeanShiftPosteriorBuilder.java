/*
    BART - Bayesian Additive Regressive Trees
    Software for Supervised Statistical Learning
    
    Copyright (C) 2012 Professor Ed George & Adam Kapelner, 
    Dept of Statistics, The Wharton School of the University of Pennsylvania

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details:
    
    http://www.gnu.org/licenses/gpl-2.0.txt

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

package CGM_BayesianCART1998;

import java.util.ArrayList;

import CGM_Statistics.*;


public class CGMRegressionMeanShiftPosteriorBuilder extends CGMPosteriorBuilder {


	/** the responses (for the data matrix) */
	private double[] y;
	/** the four hyperparameters (initially null, then cached) */
	protected Double hyper_nu;
	protected Double hyper_mu_bar;
	protected Double hyper_lambda;
	protected Double hyper_a;	
	
	public CGMRegressionMeanShiftPosteriorBuilder(CGMTreePriorBuilder tree_prior_builder) {
		super(tree_prior_builder);
	}	
	
	public CGMRegressionMeanShiftPosteriorBuilder(CGMTreePriorBuilder tree_prior_builder, double[] y) {
		super(tree_prior_builder);
		this.y = y;
//		System.out.println("instantiate CGMRegressionMeanShiftPosteriorBuilder");
	}
	
	protected double calculateLnProbYGivenTree(CGMTreeNode T) {
//		System.out.println("calculateLnProbYGivenTree CGMRegressionMeanShiftPosteriorBuilder");
//		System.out.println("calculateLnProbYGivenTree for CGMRegressionMeanShiftPosteriorBuilder");
		//get all terminal nodes and save its size
		ArrayList<CGMTreeNode> terminal_nodes = CGMTreeNode.getTerminalNodesWithDataAboveN(T, 0);
		//we'll first need to calculate the hyperparameters
		calculateHyperparameters();		
		
		int b = terminal_nodes.size();
		//calculate the denom term: the one with the product
		double denom_term = 1;
		for (int i = 0; i < b; i++){
			//get the terminal node we are looking at and cache its size
			CGMTreeNode node = terminal_nodes.get(i);
			//get number of data points in this node
			int n_i = node.data.size();
			denom_term *= Math.sqrt(n_i + hyper_a);
			
		}
		//calculate the sum term
		double sum_term = 0;
		for (int i = 0; i < b; i++){
			//get the terminal node we are looking at and cache its size
			CGMTreeNode node = terminal_nodes.get(i);
			//get number of data points in this node
			int n_i = node.data.size();
			double[] y_is = node.get_ys_in_data();
			double y_bar_i = StatToolbox.sample_average(y_is);
			double t_i = n_i * hyper_a / (n_i + hyper_a) * Math.pow(y_bar_i - hyper_mu_bar, 2);
//			System.out.println(" s_i:" + StatToolbox.sample_sum_sq_err(y_is));
//			System.out.println(" t_i:" + t_i);
			sum_term += StatToolbox.sample_sum_sq_err(y_is) + t_i + hyper_nu * hyper_lambda;
		}		
		int n = y.length;
		//just like in eq 11 of p 939 in CGM 98...
//		System.out.println("calculateLnProbYGivenTree n = " + n + " b = " + b + " ln_hyper_a = " +  Math.log(hyper_a) + " ln denom = " + Math.log(denom_term) + " Math.log(sum_term) = " + Math.log(sum_term));
		double ln_prob = b / 2.0 * Math.log(hyper_a) - Math.log(denom_term) - (n + hyper_nu) / 2.0 * Math.log(sum_term + hyper_nu * hyper_lambda);
		if (Double.compare(ln_prob, Double.NaN) == 0){
			System.err.println("ln_prob_y_given_t is NaN");
			System.exit(0);
		}
		return ln_prob;
	}

	private void calculateHyperparameters() {		
		//cache all the hyperparameters
		if (hyper_nu == null && hyper_lambda == null && hyper_mu_bar == null && hyper_a == null){
			System.out.println("calculateHyperparameters CGMRegressionMeanShiftPosteriorBuilder");
			//immediately via empirical Bayes, we set mu_bar equal to the sample average of y
			hyper_mu_bar = StatToolbox.sample_average(y);
			//now max of the sigma distribution is going to just be the se(y1,...,yn)
			double max_s = StatToolbox.sample_standard_deviation(y);
			//the minimum can be calculated by this function
			double min_s = find_min_s(max_s);
			//once we have the max and min standard deviation, we can choose nu and lambda
			calculate_nu_and_lambda(Math.pow(min_s, 2), Math.pow(max_s, 2));
//			hyper_nu = 3.0;
//			hyper_lambda = 3.0;			
			//once we have the density of sigsq, we can calculate a
			hyper_a = 1 / 3.0;
//			calculate_a();			

//			hyper_a = 1.0;
			System.out.println("hyperparams:  nu=" + hyper_nu + " lambda=" + hyper_lambda + " mu_bar=" + hyper_mu_bar + " a=" + hyper_a);
//			System.exit(0);
		}
	}

//	private void calculate_a() {
//		//which is bigger? |min_y - ybar| or |max_y - ybar|? (mu_bar = y_bar so use that)
//		double max_y_distance = Math.max(Math.abs(StatToolbox.sample_minimum(y) - hyper_mu_bar), Math.abs(StatToolbox.sample_maximum(y) - hyper_mu_bar));
//		double z_crit = StatToolbox.inv_norm_dist((1 + MostOfTheDistribution) / 2.0);
//		//now, max_dist / sigma = z_crit, solve for sigma = max_dist / z_crit
//		double sigma_sq = Math.pow(max_y_distance / z_crit, 2);
//		//now we can solve for a since sigma^2_max = 97.5th%ile of the distribution 
//		//we can get avg_sig_sq by taking the mean of the posterior inverse gamma which is nu * lambda / (nu - 2)
//		double avg_sig_sq = hyper_nu * hyper_lambda / (hyper_nu - 2);
//		hyper_a = avg_sig_sq / sigma_sq;	
//		System.out.println("max_sig_sq = " + avg_sig_sq + " max_y_distance = " + max_y_distance + " sigma^2 = " + sigma_sq + " z_crit = " + z_crit);
//	}
	
	/**
	 * Our goal is to compute the hyperparameters \nu and \lambda.
	 * We do this by assigning "most" (M) of the inverse gamma distribution between
	 * min_s and max_s. Usually M = 95% but we allow it flexibility to vary like
	 * good programmers
	 * 
	 * Now, the CDF of the inverse gamma distribution is just:
	 * 
	 * F(x) = Q(alpha, beta / x) 
	 * 
	 * where Q is the regularaized incomplete gamma function 
	 * (http://en.wikipedia.org/wiki/Incomplete_gamma_function)
	 * 
	 * Therefore we need to solve this system of equations for \alpha and \beta:
	 * 
	 * (1-M) / 2 = Q(alpha, beta / min_s) 
	 * (1+M) / 2 = Q(alpha, beta / max_s) 
	 * 
	 * Once we have these parameters, we know via equation 10 p 939 that:
	 * 
	 * \sigsq ~ InvGamma(nu / 2, nu * lambda / 2)
	 * 
	 * Hence we can solve simply for nu and lambda via:
	 * 
	 * nu = 2 * alpha
	 * lambda = beta / alpha
	 *  
	 * @param min_s_sq
	 * @param max_s_sq
	 */
	private static final double MIN_HYPER_NU = 0.001;
	private static final double MAX_HYPER_NU = 5;	
	private void calculate_nu_and_lambda(double min_s_sq, double max_s_sq) {
		System.out.println("calculate_nu_and_lambda CGMRegressionMeanShiftPosteriorBuilder min_s_sq = " + min_s_sq + " max_s_sq = " + max_s_sq);
//		double left_prob_bound = (1 - MostOfTheDistribution) / 2;
//		double right_prob_bound = (1 + MostOfTheDistribution) / 2;
		
		//go through a grid and just take the maximum probability
		double min_prob_diff = Double.MAX_VALUE;
		for (double nu = MIN_HYPER_NU; nu <= MAX_HYPER_NU; nu += MAX_HYPER_NU / 100.0){
			for (double lambda = max_s_sq / 1000.0; lambda < max_s_sq; lambda += max_s_sq / 1000.0){
				//remember the probability is F(s_max^2) - F(s_min^2)
				double prob = StatToolbox.cumul_dens_function_inv_gamma(nu / 2.0, nu * lambda / 2.0, min_s_sq, max_s_sq);
				double prob_diff = Math.abs(prob - CGMShared.MostOfTheDistribution);
				if (prob_diff < min_prob_diff){
					min_prob_diff = prob_diff;
					//record the values
					hyper_nu = nu;
					hyper_lambda = lambda;
//					System.out.println("converging nu and lambda prob = " + Math.round(prob * 100) + "% nu = " + nu + " lambda = " + lambda + " min_s_sq = " + min_s_sq + " max_s_sq = " + max_s_sq);
				}
			}
		}		
	}

	private static final int MIN_S_CONST_FACTOR = 10; //letting max R^2 = 99%
	private double find_min_s(double max_s) {
		//this is remarkably simple, and we should work on revamping this, but we will for now just return a third of the max
		return max_s / MIN_S_CONST_FACTOR;
	}

}
