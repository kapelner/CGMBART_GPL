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
public class CGMBARTPosteriorBuilder {

	private double hyper_sigsq_mu;
	private double sigsq;
	private double hyper_mu_bar;
	private CGMTreePriorBuilder tree_prior_builder;
	
	
	public CGMBARTPosteriorBuilder(CGMTreePriorBuilder tree_prior_builder) {
		this.tree_prior_builder = tree_prior_builder;
	}
	
	public void setHyperparameters(double hyper_mu_bar, double hyper_sigsq_mu){
		this.hyper_mu_bar = hyper_mu_bar;
		this.hyper_sigsq_mu = hyper_sigsq_mu;
	}
	
	public enum Steps {GROW, PRUNE};
	/**
	 * Iterates the Metropolis-Hastings algorithm by one step
	 * This is a public method because it gets called in the BART implementation
	 * 
	 * @param T_i				the original tree
	 * @param iteration_name	we just use this when naming the image file of this illustration
	 * @return 					the next tree (T_{i+1}) via one iteration of M-H
	 */
	public CGMTreeNode iterateMHPosteriorTreeSpaceSearch(CGMTreeNode T_i) {
		final CGMTreeNode T_star = T_i.clone(true);
		//each proposal will calculate its own value, but this has to be initialized atop		
		double log_r = 0;
		switch (randomlyPickAmongTheFourProposalSteps(T_i)){
			case GROW:
				CGMTreeNode grow_node = pickGrowNode(T_star);
				//if we can't grow, reject offhand
				if (grow_node == null){
					System.out.println("proposal REJECTED DUE TO CANNOT GROW\n\n");
					return T_i;					
				}
				double ln_transition_ratio_grow = calcLnTransRatioGrow(T_i, T_star, grow_node);
				double ln_likelihood_ratio_grow = calcLnLikRatioGrow(grow_node);
				double ln_tree_structure_ratio_grow = calcLnTreeStructureRatioGrow(T_i, T_star, grow_node);
				log_r = ln_transition_ratio_grow + ln_likelihood_ratio_grow + ln_tree_structure_ratio_grow;
				break;
			case PRUNE:
				CGMTreeNode prune_node = pickPruneNode(T_star);
				//if we can't grow, reject offhand
				if (prune_node == null){
					System.out.println("proposal REJECTED DUE TO CANNOT PRUNE\n\n");
					return T_i;					
				}
				double ln_transition_ratio_prune = calcLnTransRatioGrow(T_i, T_star, grow_node);
				double ln_likelihood_ratio_prune = calcLnLikRatioGrow(grow_node);
				double ln_tree_structure_ratio_prune = calcLnTreeStructureRatioGrow(T_i, T_star, grow_node);
				log_r = ln_transition_ratio_prune + ln_likelihood_ratio_prune + ln_tree_structure_ratio_prune;
				break;
		}		
		double ln_u_0_1 = Math.log(Math.random());
		//ACCEPT/REJECT,STEP_name,log_prop_lik_o,log_prop_lik_f,log_r 
		CGMBART.mh_iterations_full_record.println(
			(acceptProposal(ln_u_0_1, log_r) ? "A" : "R") + "," +  
			TreeIllustration.one_digit_format.format(log_r) + "," +
			TreeIllustration.two_digit_format.format(ln_u_0_1)
		);		
//		System.out.println("ln_u_0_1: " + ln_u_0_1 + " ln_r: " + log_r);
		if (acceptProposal(ln_u_0_1, log_r)){ //accept proposal
			System.out.println("proposal ACCEPTED\n\n");
			return T_star;
		}
		//reject proposal
		System.out.println("proposal REJECTED\n\n");
		return T_i;
	}
	
	protected boolean acceptProposal(double ln_u_0_1, double log_r){
		return ln_u_0_1 < log_r ? true : false;
	}
	
//	private static final double ln_one_over_sqrt_two_pi = Math.log(1 / Math.sqrt(2 * Math.PI));
//	private static final double ln_two_pi = Math.log(2 * Math.PI);

	/**
	 * This calculates the marginalized ln(P(R_1, \ldots, R_N | \sigsq))
	 * by calculating g(\Rbar | \sigsq)
	 * 
	 * @param node
	 * @return
	 */
//	private double calculcateLnPropProbOfNode(CGMTreeNode node) {
//		double log_prob_node_ell = 0;
//		log_prob_node_ell -= 0.5 * (Math.log(hyper_sigsq_mu) + Math.log(node.n / sigsq + 1));
//		log_prob_node_ell += 
//		
//		DoubleMatrix rs = new DoubleMatrix(node.get_ys_in_data());
//		DoubleMatrix mu = new DoubleMatrix(hyper_mu_bar, node.n);
//		
//		CGMBART.mh_iterations_full_record.print(
//			TreeIllustration.two_digit_format.format(log_prob_node_ell) + ","
//		);		
//		//cache this for further use
//		node.log_prop_lik = log_prob_node_ell;
//		
//		return log_prob_node_ell;
//	}
	/** The number of data points in a node that we can split on */
	protected static final int N_RULE = 5;	

	/** This grow step is different
	 * 
	 */
	protected CGMTreeNode pickGrowNode(CGMTreeNode T) {
//		System.out.println("proposal via GROW  " + T.stringID());
		//find all terminals nodes that have **more** than N_RULE data 
		ArrayList<CGMTreeNode> growth_nodes = CGMTreeNode.getTerminalNodesWithDataAboveN(T, N_RULE);
//		System.out.print("num growth nodes: " + growth_nodes.size() +":");
//		for (CGMTreeNode node : growth_nodes){
//			System.out.print(" " + node.stringID());
//		}
//		System.out.print("\n");
		
		//2 checks
		//a) If there is no nodes to grow, return null
		//b) If the node we picked CANNOT grow due to no available predictors, return null as well
		//we return a probability of null
		if (growth_nodes.size() == 0){
			System.out.println("no growth nodes in GROW step!");
			return null;
		}		
		
		//now we pick one of these nodes with enough data points randomly
		CGMTreeNode growth_node = growth_nodes.get((int)Math.floor(Math.random() * growth_nodes.size()));
		
		//if we can't grow at all, return null
		if (growth_node.cannotGrow()){
			System.out.println("cannot grow node in GROW step!");
			return null;			
		}
		return growth_node;
	}
	
	protected Steps randomlyPickAmongTheFourProposalSteps(CGMTreeNode T) {
		//the first thing we need to check is if we can prune
		if (T.canBePruned()){
			double roll = Math.random();
			if (roll < 0.5)
				return Steps.GROW;
			return Steps.PRUNE;			
		}
		return Steps.GROW;
	}	

	public void setCurrentSigsqValue(double sigsq) {
		this.sigsq = sigsq;
	}

}
