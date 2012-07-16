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

import org.apache.commons.math.special.Gamma;

import CGM_Statistics.*;


public final class CGMClassificationPosteriorBuilder extends CGMPosteriorBuilder {

	/** the number of classes */
	private int K;
	/** cache this calculation constant to make algorithm faster */
	private Double alpha_constant;

	public CGMClassificationPosteriorBuilder(CGMTreePriorBuilder tree_prior_builder, int K) {
		super(tree_prior_builder);
		this.K = K;
	}
	
	protected double calculateLnProbYGivenTree(CGMTreeNode T) {
//		System.out.println("calculateLnProbYGivenTree  node:" + T.stringID() + " leaf:" + T.isLeaf + " left: " + T.left + " right:" + T.right);		
		
//		System.out.println("calculateProbYGivenTree");
		//get all terminal nodes and save its size
		ArrayList<CGMTreeNode> terminal_nodes = CGMTreeNode.getTerminalNodesWithDataAboveOrEqualToN(T, 0);
		int b = terminal_nodes.size();
		//first get the constants
		double b_times_alpha_const = b * calculateAlphaConstant();
//		System.out.println("alpha_constant_to_the_b: " + alpha_constant_to_the_b);
		//now calculate the big kahuna term
		double big_kahuna = 0;
		//loops explained:
		//i is over the nodes, 
		//j is over the data records in each node, 
		//k is over the phenotypes {C_1, \ldots, C_k, \ldots, C_K}
		for (int i = 0; i < b; i++){
			//get the terminal node we are looking at and cache its size
			CGMTreeNode node = terminal_nodes.get(i);
			int n_i = node.data.size();
//			System.out.println("i: " + i + " node: " + node.stringID() + " n_i: " + n_i);
			
			//calculate denom first
			double second_term = Gamma.logGamma(n_i + getSumAlphas());
			
			//now if denom is too big (i.e. infinity), we're going to bust out and
			//claim that this probability is zilch and force the tree to grow
//			if (Double.isInfinite(second_term)){
//				System.out.println("denom INF");
//				return 0.00000001 * b; //just weight by the relative size of tree
//			}
			
//			System.out.println("denom: " + StatToolbox.gammaFunction(n_i + sum_alphas));
			//count the buckets for each phenotype
			int[] n_ik = new int[K];
			for (int j = 0; j < n_i; j++){
				//pick out the y out of the jth record and record it
				n_ik[(int)node.data.get(j)[treePriorBuilder.getP()]]++;
			}	
			//calculate the numerator portion
			double first_term = 0;
			for (int k = 0; k < K; k++){
//				System.out.println("k: " + k + " count + alpha: " + (n_ik[k] + treePriorBuilder.getAlphaForParameterPrior(k)));
				first_term += Gamma.logGamma(n_ik[k] + AlphaHyperParams[k]);
//				System.out.println("numer *= " + StatToolbox.gammaFunction(n_ik[k] + treePriorBuilder.getAlphaForParameterPrior(k)));
			}
//			System.out.println("numer: " + numer);
			//calculating the denominator is easy
			

//			System.out.println("big_kahuna *= " + (numer / denom));
			big_kahuna += (first_term - second_term);
		}
//		System.out.println("ln_prob_y_given_T: " + (b_times_alpha_const + big_kahuna));
		return b_times_alpha_const + big_kahuna;
	}	
	

	/** the alpha hyperparams as a fixed vector, change this value if needed */
	public static final double[] AlphaHyperParams = new double[1000];
	static {
		for (int i = 0; i < 1000; i++){
			AlphaHyperParams[i] = 1;
		}
	}
	
	public double getSumAlphas(){
		double sum_alphas = 0;
		for (int k = 0; k < K; k++){
			sum_alphas += AlphaHyperParams[k];
		}
		return sum_alphas;
	}	
	
	/** this calculates the alpha constant and caches it (see paper) */
	private double calculateAlphaConstant() {
		if (alpha_constant == null){
//			System.out.println("\n\ncalculateAlphaConstant");
//			System.out.println("getSumAlphas: " + treePriorBuilder.getSumAlphas());
//			System.out.println("numer: " + StatToolbox.gammaFunction(treePriorBuilder.getSumAlphas()));
			//then calculate it:
			double sum_lngamma_alphas = 0;			
			for (int k = 0; k < K; k++){
				sum_lngamma_alphas += Gamma.logGamma(AlphaHyperParams[k]);
//				System.out.println("k: " + k + "  prod_gamma_alphas: " + prod_gamma_alphas);
			}			
			alpha_constant = Gamma.logGamma(getSumAlphas()) - sum_lngamma_alphas;
			System.out.println("alpha_constant: " + alpha_constant + "\n");
		}
		return alpha_constant;
	}

}
