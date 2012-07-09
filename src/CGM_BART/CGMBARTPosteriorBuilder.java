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
import java.util.HashSet;

import CGM_Statistics.*;

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
	private CGMBARTPriorBuilder tree_prior_builder;
	
	
	public CGMBARTPosteriorBuilder(CGMBARTPriorBuilder tree_prior_builder) {
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
		switch (randomlyPickAmongTheTwoProposalSteps(T_i)){
			case GROW:
				log_r = doMHGrowAndCalcLnR(T_i, T_star);
				break;
			case PRUNE:
				log_r = doMHPruneAndCalcLnR(T_i, T_star);
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

	private double doMHGrowAndCalcLnR(CGMTreeNode T_i, CGMTreeNode T_star) {
		CGMTreeNode grow_node = pickGrowNode(T_star);
		//if we can't grow, reject offhand
		if (grow_node == null){
			System.out.println("proposal ln(r) = -oo DUE TO CANNOT GROW\n\n");
			return Double.NEGATIVE_INFINITY;					
		}
		growNode(grow_node);
		double ln_transition_ratio_grow = calcLnTransRatioGrow(T_i, T_star, grow_node);
		double ln_likelihood_ratio_grow = calcLnLikRatioGrow(grow_node);
		double ln_tree_structure_ratio_grow = calcLnTreeStructureRatioGrow(grow_node);
		
		return ln_transition_ratio_grow + ln_likelihood_ratio_grow + ln_tree_structure_ratio_grow;
	}
	
	private double doMHPruneAndCalcLnR(CGMTreeNode T_i, CGMTreeNode T_star) {
		CGMTreeNode prune_node = pickPruneNode(T_star);
		//if we can't grow, reject offhand
		if (prune_node == null){
			System.out.println("proposal ln(r) = -oo DUE TO CANNOT PRUNE\n\n");
			return Double.NEGATIVE_INFINITY;					
		}				
		double ln_transition_ratio_prune = calcLnTransRatioPrune(T_i, T_star, prune_node);
		double ln_likelihood_ratio_prune = Math.pow(calcLnLikRatioGrow(prune_node), -1); //inverse of before (will speed up later)
		double ln_tree_structure_ratio_prune = calcLnTreeStructureRatioPrune(prune_node);
		pruneNode(prune_node);
		return ln_transition_ratio_prune + ln_likelihood_ratio_prune + ln_tree_structure_ratio_prune;
	}	

	private void pruneNode(CGMTreeNode prune_node) {
		// TODO Auto-generated method stub
		
	}

	private void growNode(CGMTreeNode node) {
		//we already assume the node can grow, now we just have to pick an attribute and split point
		node.splitAttributeM = node.pickRandomPredictorThatCanBeAssigned();
		node.possible_split_values = tree_prior_builder.possibleSplitValues(node);
		node.splitValue = node.pickRandomSplitValue();
		//split the data correctly
		ClassificationAndRegressionTree.SortAtAttribute(node.data, node.splitAttributeM);
		int n_split = ClassificationAndRegressionTree.getSplitPoint(node.data, node.splitAttributeM, node.splitValue);
		//now build the node offspring
		node.left = new CGMTreeNode(node, ClassificationAndRegressionTree.getLowerPortion(node.data, n_split));
		node.right = new CGMTreeNode(node, ClassificationAndRegressionTree.getUpperPortion(node.data, n_split));		
	}

	private double calcLnTreeStructureRatioPrune(CGMTreeNode T_i) {
		// TODO Auto-generated method stub
		return 0;
	}

	private double calcLnTransRatioPrune(CGMTreeNode T_i, CGMTreeNode T_star, CGMTreeNode prune_node) {
		// TODO Auto-generated method stub
		return 0;
	}

	private CGMTreeNode pickPruneNode(CGMTreeNode T_star) {
		// TODO Auto-generated method stub
		return null;
	}

	private double calcLnTreeStructureRatioGrow(CGMTreeNode grow_node) {
		double alpha = tree_prior_builder.getAlpha();
		double beta = tree_prior_builder.getBeta();
		int d_eta = grow_node.generation;
		int p_eta = grow_node.pAdj();
		int n_eta = grow_node.nAdj();
		int p_eta_L = grow_node.left.pAdj();
		int n_eta_L = grow_node.left.nAdj();
		int p_eta_R = grow_node.right.pAdj();
		int n_eta_R = grow_node.right.nAdj();
		return Math.log(alpha) 
				+ beta * Math.log(1 + d_eta) - 2 * beta * Math.log(2 + d_eta)
				+ Math.log(p_eta) + Math.log(n_eta)
				- Math.log(p_eta_L) - Math.log(n_eta_L) - Math.log(p_eta_R) - Math.log(n_eta_R);
	}

	private double calcLnLikRatioGrow(CGMTreeNode grow_node) {
		int n_ell = grow_node.getN();
		int n_ell_L = grow_node.left.getN();
		int n_ell_R = grow_node.right.getN();
		System.out.println("calcLnTransRatioGrow n_ell: " + n_ell + " n_ell_L: " + n_ell_L + " n_ell_R: " + n_ell_R);
		double sigsq_plus_n_ell_hyper_sisgsq_mu = sigsq + n_ell * hyper_sigsq_mu;
		double sigsq_plus_n_ell_L_hyper_sisgsq_mu = sigsq + n_ell_L * hyper_sigsq_mu;
		double sigsq_plus_n_ell_R_hyper_sisgsq_mu = sigsq + n_ell_R * hyper_sigsq_mu;
		//now go ahead and calculate it out		
		double c = 0.5 * (
				Math.log(sigsq) 
				+ Math.log(sigsq_plus_n_ell_hyper_sisgsq_mu) 
				- Math.log(sigsq_plus_n_ell_L_hyper_sisgsq_mu) 
				- Math.log(sigsq_plus_n_ell_R_hyper_sisgsq_mu));
		double d = hyper_sigsq_mu / (2 * sigsq);
		double e = grow_node.left.sumResponsesSqd() / sigsq_plus_n_ell_L_hyper_sisgsq_mu
				+ grow_node.right.sumResponsesSqd() / sigsq_plus_n_ell_R_hyper_sisgsq_mu
				- grow_node.sumResponsesSqd() / sigsq_plus_n_ell_hyper_sisgsq_mu;
		
		return c + d * e;
	}

	private double calcLnTransRatioGrow(CGMTreeNode T_i, CGMTreeNode T_star, CGMTreeNode grow_node) {
		int b = T_i.numLeaves();
		int p_adj = grow_node.pAdj();
		int n_adj = grow_node.nAdj();
		int w_2_star = T_star.numPruneNodesAvailable();
		return Math.log(b) + Math.log(p_adj) + Math.log(n_adj) - Math.log(w_2_star); 
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

	protected CGMTreeNode pickGrowNode(CGMTreeNode T) {
		ArrayList<CGMTreeNode> growth_nodes = CGMTreeNode.getTerminalNodesWithDataAboveN(T, N_RULE);
		
		//2 checks
		//a) If there is no nodes to grow, return null
		//b) If the node we picked CANNOT grow due to no available predictors, return null as well
		
		//do check a
		if (growth_nodes.size() == 0){
			System.out.println("no growth nodes in GROW step!");
			return null;
		}		
		
		//now we pick one of these nodes with enough data points randomly
		CGMTreeNode growth_node = growth_nodes.get((int)Math.floor(Math.random() * growth_nodes.size()));
		//find which predictors can be used
		growth_node.predictors_that_can_be_assigned = tree_prior_builder.predictorsThatCouldBeUsedToSplitAtNode(growth_node);
		//do check b
		if (growth_node.pAdj() == 0){
			System.out.println("no attributes available in GROW step!");
			return null;			
		}
		//if we passed, we can use this node
		return growth_node;
	}
	
	private boolean cannotPrune(CGMTreeNode T) {
		return T.parent == null;
	}
	
	protected Steps randomlyPickAmongTheTwoProposalSteps(CGMTreeNode T) {
		//the first thing we need to check is if we can prune
		if (!cannotPrune(T)){
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
