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

package CGM_BART;

import java.util.ArrayList;

import CGM_BayesianCART1998.*;
import CGM_Statistics.*;
import GemIdentTools.Matrices.DoubleMatrix;

/**
 * This is the same thing as CGMRegressionMeanShiftPosteriorBuilder
 * except we are feedings the hyperparameters to it
 * that way it will not calculate them using the CGM98 methods
 * 
 * @author kapelner
 *
 */
public final class CGMBARTPosteriorBuilder extends CGMRegressionMeanShiftPosteriorBuilder {

	private double hyper_sigsq_mu;
	private double sigsq;
	
	
	public CGMBARTPosteriorBuilder(CGMTreePriorBuilder tree_prior_builder) {
		super(tree_prior_builder);
	}		

	public CGMBARTPosteriorBuilder(CGMTreePriorBuilder tree_prior_builder, double[] y) {
		super(tree_prior_builder, y);
	}
	
	public void setHyperparameters(double hyper_mu_bar, double hyper_a, double hyper_nu, double hyper_lambda, double hyper_sigsq_mu){
		this.hyper_mu_bar = hyper_mu_bar;
		this.hyper_a = hyper_a;
		this.hyper_nu = hyper_nu;
		this.hyper_lambda = hyper_lambda;
		this.hyper_sigsq_mu = hyper_sigsq_mu;
	}
	
//	private static final double ln_one_over_sqrt_two_pi = Math.log(1 / Math.sqrt(2 * Math.PI));
	private static final double ln_two_pi = Math.log(2 * Math.PI);
	/**
	 * Note this is not the probability of Y given the tree, this is the probability
	 * of the r_i's that are in this tree, since each tree only fits the *remaining* data
	 */
	protected double calculateLnProbYGivenTree(CGMTreeNode T) {
//		System.out.println("hyper_sigsq_mu: " + hyper_sigsq_mu);
		
		//first get the leaves
		ArrayList<CGMTreeNode> terminal_nodes = CGMTreeNode.getTerminalNodesWithDataAboveN(T, 0);
//		int b = terminal_nodes.size();
		int n = treePriorBuilder.getN();
		
		//calculate term zero:
		double ln_prob = -n * ln_two_pi; //- 0.5 * (n - b) * Math.log(sigsq);
//		System.out.println("calculateLnProbYGivenTree after term 0: " + ln_prob);
		
		//now loop over all nodes and calculate term a
		double term_a = 0;
		for (CGMTreeNode node : terminal_nodes){
//			DoubleMatrix sigma = generateSigmaMatrix(node.n);
//			System.out.println("sigma: \n" + sigma.crop(0, 5, 0, 5).toString(4));
//			double det = sigma.det();
//			System.out.println("det(sigma) " + det);
			double det_by_book = Math.pow(sigsq, node.n - 1) * (hyper_sigsq_mu * node.n + sigsq);
			//do some corrections to the determinant due to computational limitations of the double storage
			if (det_by_book == 0){
				det_by_book = Double.MIN_NORMAL;
			}
//			if (det_by_book == Double.NaN || det_by_book == Double.NEGATIVE_INFINITY || det_by_book == Double.POSITIVE_INFINITY){
//				System.err.println("det: " + det_by_book + " sigsq: " + sigsq + " sigsqmu: " + hyper_sigsq_mu + " n: " + node.n);
//			}
			if (det_by_book == Double.POSITIVE_INFINITY){
				det_by_book = Double.MAX_VALUE;
			}
//			System.out.println("det(sigma) as calculated with book " + det_by_book + " ln(det(sigma)) as calculated with book " + Math.log(det_by_book));			
			term_a += Math.log(det_by_book);
//			ln_prob -= 0.5 * Math.log(det_by_book);
//			System.out.println("calculateLnProbYGivenTree running total: " + ln_prob);
		}	
		ln_prob -= 0.5 * term_a;
		
		
		//now loop over all nodes and calculate term b
		double term_b = 0;		
		for (CGMTreeNode node : terminal_nodes){
			DoubleMatrix rs = new DoubleMatrix(node.get_ys_in_data());
			DoubleMatrix mu = new DoubleMatrix(hyper_mu_bar, node.n);
			DoubleMatrix rs_min_mu = rs.minus(mu);
//			System.out.println("rs dims " + rs.getRowDimension() + " x " + rs.getColumnDimension() + " rs: \n" + rs.transpose().toString(2));
//			System.out.println("mu dims " + mu.getRowDimension() + " x " + mu.getColumnDimension() + " mu: \n" + mu.transpose().toString(2));
//			System.out.println("rs_min_mu dims " + rs_min_mu.getRowDimension() + " x " + rs_min_mu.getColumnDimension() + " rs_min_mu: \n" + rs_min_mu.transpose().toString(2));
//			System.out.println("rs transpose dims " + rs.transpose().getRowDimension() + " x " + rs.transpose().getColumnDimension() + "  " + rs.transpose().toString(1));
//			DoubleMatrix sigma = generateSigmaMatrix(node.n);
			DoubleMatrix inv_sigma = generateInvSigmaMatrix(node.n);
//			System.out.println("inv_sigma dims " + inv_sigma.getRowDimension() + " x " + inv_sigma.getColumnDimension());
//			System.out.println("sigma_inv \n " + generateSigmaMatrix(node.n).inverse().crop(0, 5, 0, 5).toString(2));
//			System.out.println("sigma_inv as calculated with book \n" + inv_sigma.crop(0, 5, 0, 5).toString(2));
			
//			System.out.println("sigma_matrix dims " + sigma_matrix.getRowDimension() + " x " + sigma_matrix.getColumnDimension());
//			System.out.println("inv_sigma_times_rs dims " + inv_sigma.times(rs).getRowDimension() + " x " + inv_sigma.times(rs).getColumnDimension());
			DoubleMatrix main_term = rs_min_mu.transpose().times(inv_sigma.times(rs_min_mu));
//			System.out.println("main_term: " + main_term.get(0, 0));
			term_b += main_term.get(0, 0); //it should be a scalar
//			ln_prob -= 0.5 * Math.log(main_term.get(0, 0));
//			System.out.println("calculateLnProbYGivenTree running total: " + ln_prob);			
		}
		ln_prob -= 0.5 * term_b;
		System.out.println("calculateLnProbYGivenTree is prop to: " + ln_prob);
//		System.out.println("ProbYGivenTree :" + Math.pow(ln_prob, Math.E));
		
		return ln_prob;
	}

	private DoubleMatrix generateInvSigmaMatrix(int n) {
		DoubleMatrix JnOverN = DoubleMatrix.JnOverN(n);
		DoubleMatrix In = DoubleMatrix.In(n);	
		DoubleMatrix first_term = JnOverN.times(1 / (hyper_sigsq_mu * n + sigsq));
		DoubleMatrix second_term = (In.minus(JnOverN)).times(1 / sigsq);
		return first_term.plus(second_term);
	}

//	private DoubleMatrix generateSigmaMatrix(int n) {
//		DoubleMatrix sigma = new DoubleMatrix(n, n);
//		for (int i = 0; i < n; i++){
//			for (int j = 0; j < n; j++){
//				if (i == j){
//					sigma.set(i, j, hyper_sigsq_mu + sigsq);
//				}
//				else {
//					sigma.set(i, j, hyper_sigsq_mu);
//				}
//			}
//		}
//		return sigma;
//	}
//
//	private double calculate_sse_from_leaf_data(double[] rs) {
//		double sse = 0;
//		for (double r : rs){
//			sse += Math.pow(r, 2);
//		}
//		return sse;
//	}

	/** 
	 * we also make sure we use the new distribution of tree-space moves
	 */
	protected Steps randomlyPickAmongTheFourProposalSteps() {
		double roll = Math.random();
		if (roll < 0.25)
			return Steps.GROW;
		else if (roll < 0.5)
			return Steps.PRUNE;
		else if (roll < 0.9)
			return Steps.CHANGE;
		return Steps.SWAP;
	}

	public void setCurrentSigsqValue(double sigsq) {
//		System.out.println("setCurrentSigsqValue sigsq = " + sigsq + " sigsq_mu = " + hyper_sigsq_mu);
		this.sigsq = sigsq;
	}

}
