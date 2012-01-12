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

import java.util.List;

import CGM_Statistics.*;

import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentView.JProgressBarAndLabel;

public class CGMClassificationTree extends CGMCART implements LeafAssigner {
	private static final long serialVersionUID = 5178072095746281246L;
	/** the number of classes in this tree (set by user) */
	private int K;
	
	public CGMClassificationTree(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress, int K) {
		super(datumSetupForEntireRun, buildProgress);
		this.K = K;
		
	}

	@Override
	protected void createPosteriorBuilder() {
		tree_posterior_builder = new CGMClassificationPosteriorBuilder(tree_prior_builder, K);	
	}		

	/**
	 * For each leaf $i$, we need to assign a class, $C_k$ which we
	 * represent by $\theta_i \in \braces{C_1, \ldots, C_K}$. In order to do
	 * that, we take the mode of the posterior distribution of $\theta_i$ given
	 * the distribution of $y$'s in the node which is a dirichlet (since dirichlet
	 * is the conjugate for the multinomial which was the initial assumption)
	 * 
	 * $p_{i1}, \ldots, p_{iK} | \bv{n_i}, T &\sim& \dirichlet{\alphavec + \bv{n_i}}$
	 * 
	 * Therefore we need to do the following:
	 * 
	 * a) generate the $p_{i1}, \ldots, p_{iK}$ using the posterior Dirichlet
	 * b) sample one point from the multinomial
	 * 
	 * @param leaf		the leaf node to assign the class value to
	 */
	public void assignLeaf(CGMTreeNode leaf) {
		int[] ns = countCompartments(leaf.data);
		leaf.klass = (double)StatToolbox.FindMaxIndex(ns);
//		double[] ps = generateProbDistOfClassProbsAccdToPosteriorDirichlet(ns);
//		Integer klass = posterior_mode(ps);
////		System.out.println("class: " + klass);
//		leaf.klass = (double)klass;
	}
	
	private int[] countCompartments(List<double[]> data) {
		int[] counts = new int[K];
		for (double[] record : data){
			counts[(int)record[p]]++; //record[p] is the y value
		}
		return counts;
	}	
	///// all functions related to parameter prior ($\thetavec$)

	/**
	 * Find the maximum probability of this multinomial distribution
	 * 
	 * @param ps	the probabilities for each class
	 * @return
	 */
//	private Integer posterior_mode(double[] ps){
//		double max_prob = Double.MIN_VALUE;
//		Integer max_index = null;
//		for (int k = 0; k < K; k++){
//			if (ps[k] > max_prob){
//				max_prob = ps[k];
//				max_index = k;
//			}
//		}
//		return max_index;
//	}
	
	/**
	 * 
	 * @param ns
	 * @return
	 */
//	private double[] generateProbDistOfClassProbsAccdToPosteriorDirichlet(int[] ns){
//		//sample the p's from the Dirichlet
//		double[] post_gammas = new double[K];
//		double sum_post_gammas = 0;
//		//generate gammas according to Wikipedia's page on 
//		//simulating Dirichlets	
////		System.out.print("post gammas: ");
//		for (int k = 0; k < K; k++){
//			post_gammas[k] = StatToolbox.generate_gamma(CGMClassificationPosteriorBuilder.AlphaHyperParams[k] + ns[k], 1);
//			sum_post_gammas += post_gammas[k];
////			System.out.print(post_gammas[k] + ", ");
//		}
////		System.out.print("\nps: ");
//		double[] ps = new double[K];
//		for (int k = 0; k < K; k++){
//			ps[k] = post_gammas[k] / sum_post_gammas;
////			System.out.print(ps[k] + ", ");
//		}
////		System.out.print("\n");
//		return ps;
//	}

	
}
