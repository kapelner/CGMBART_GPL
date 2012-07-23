package CGM_BART;

import java.io.Serializable;
import java.util.ArrayList;

import CGM_Statistics.StatToolbox;
import CGM_Statistics.TreeIllustration;

public abstract class CGMBART_mh extends CGMBART_gibbs implements Serializable {
	private static final long serialVersionUID = 1825856510284398699L;


	public enum Steps {GROW, PRUNE};
	/**
	 * Iterates the Metropolis-Hastings algorithm by one step
	 * This is a public method because it gets called in the BART implementation
	 * 
	 * @param T_i				the original tree
	 * @param iteration_name	we just use this when naming the image file of this illustration
	 * @return 					the next tree (T_{i+1}) via one iteration of M-H
	 */
	public CGMBARTTreeNode iterateMHPosteriorTreeSpaceSearch(CGMBARTTreeNode T_i) {
//		System.out.println("iterateMHPosteriorTreeSpaceSearch");
		final CGMBARTTreeNode T_star = T_i.clone(true);
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
		double ln_u_0_1 = Math.log(StatToolbox.rand());
		System.out.println("u = " + Math.exp(ln_u_0_1) + 
				" <? r = " + 
				(Math.exp(log_r) < 0.0000001 ? "damn small" : Math.exp(log_r)));
		
		//ACCEPT/REJECT,STEP_name,log_prop_lik_o,log_prop_lik_f,log_r 
		CGMBART_debug.mh_iterations_full_record.println(
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

	private double doMHGrowAndCalcLnR(CGMBARTTreeNode T_i, CGMBARTTreeNode T_star) {
		System.out.println("doMHGrowAndCalcLnR on " + T_star.stringID() + " old depth: " + T_star.deepestNode());
		CGMBARTTreeNode grow_node = pickGrowNode(T_star);
		//if we can't grow, reject offhand
		if (grow_node == null){ // || grow_node.generation >= 2
			System.out.println("proposal ln(r) = -oo DUE TO CANNOT GROW\n\n");
			return Double.NEGATIVE_INFINITY;					
		}
		growNode(grow_node);
//		System.out.println("grow_node: " + grow_node.stringID() + " new depth: " + T_star.deepestNode() + " " + grow_node.deepestNode());
		double ln_transition_ratio_grow = calcLnTransRatioGrow(T_i, T_star, grow_node);
		double ln_likelihood_ratio_grow = calcLnLikRatioGrow(grow_node);
		double ln_tree_structure_ratio_grow = calcLnTreeStructureRatioGrow(grow_node);
		
		return ln_transition_ratio_grow + ln_likelihood_ratio_grow + ln_tree_structure_ratio_grow;
	}
	
	private double doMHPruneAndCalcLnR(CGMBARTTreeNode T_i, CGMBARTTreeNode T_star) {
		System.out.println("doMHPruneAndCalcLnR");
		CGMBARTTreeNode prune_node = pickPruneNode(T_star);
		//if we can't grow, reject offhand
		if (prune_node == null){
			System.out.println("proposal ln(r) = -oo DUE TO CANNOT PRUNE\n\n");
			return Double.NEGATIVE_INFINITY;					
		}				
		double ln_transition_ratio_prune = calcLnTransRatioPrune(T_i, T_star, prune_node);
		double ln_likelihood_ratio_prune = -calcLnLikRatioGrow(prune_node); //inverse of before (will speed up later)
		double ln_tree_structure_ratio_prune = -calcLnTreeStructureRatioGrow(prune_node);
		CGMBARTTreeNode.pruneTreeAt(prune_node);
		return ln_transition_ratio_prune + ln_likelihood_ratio_prune + ln_tree_structure_ratio_prune;
	}	
	
	private CGMBARTTreeNode pickPruneNode(CGMBARTTreeNode T) {
		
		//2 checks
		//a) If this is the root, we can't prune so return null
		//b) If there are no prunable nodes (not sure how that could happen), return null as well
		
		if (T.isStump()){
			System.out.println("cannot prune a stump!");
			return null;			
		}
		
		ArrayList<CGMBARTTreeNode> prunable_nodes = CGMBARTTreeNode.getPrunableNodes(T);
		if (prunable_nodes.size() == 0){
			System.out.println("no prune nodes in PRUNE step!  T parent: " + T.parent + " T left: " + T.left + " T right: " + T.right);
			return null;
		}		
		
		//now we pick one of these nodes randomly
		return prunable_nodes.get((int)Math.floor(StatToolbox.rand() * prunable_nodes.size()));
	}
	

	private double calcLnTransRatioPrune(CGMBARTTreeNode T_i, CGMBARTTreeNode T_star, CGMBARTTreeNode prune_node) {
		int w_2 = T_i.numPruneNodesAvailable();
		int b = T_i.numLeaves();
		int p_adj = prune_node.pAdj();
		int n_adj = prune_node.nAdj();
		int n_repeat = prune_node.splitValuesRepeated();
		return Math.log(w_2) + Math.log(n_repeat) - Math.log(b - 1) - Math.log(p_adj) - Math.log(n_adj); 
	}

	private void growNode(CGMBARTTreeNode grow_node) {
		//we already assume the node can grow, now we just have to pick an attribute and split point
		grow_node.splitAttributeM = grow_node.pickRandomPredictorThatCanBeAssigned();
		grow_node.splitValue = grow_node.pickRandomSplitValue();
		grow_node.isLeaf = false;
		grow_node.left = new CGMBARTTreeNode(grow_node);
		grow_node.right = new CGMBARTTreeNode(grow_node);
		CGMBARTTreeNode.propagateRuleChangeOrSwapThroughoutTree(grow_node, true);
	}


	private double calcLnTreeStructureRatioGrow(CGMBARTTreeNode grow_node) {
		int d_eta = grow_node.generation;
		int p_adj = grow_node.pAdj();
		int n_adj = grow_node.nAdj();
		int n_repeat = grow_node.splitValuesRepeated();
		return Math.log(ALPHA) 
				+ 2 * Math.log(1 - ALPHA / Math.pow(2 + d_eta, BETA))
				+ Math.log(n_repeat)
				- Math.log(Math.pow(1 + d_eta, BETA) - ALPHA)
				- Math.log(p_adj) 
				- Math.log(n_adj);
	}

	private double calcLnLikRatioGrow(CGMBARTTreeNode grow_node) {
		double sigsq = gibbs_samples_of_sigsq.get(gibb_sample_num - 1);
		int n_ell = grow_node.getN();
		int n_ell_L = grow_node.left.getN();
		int n_ell_R = grow_node.right.getN();
		System.out.println("calcLnLikRatioGrow n_ell: " + n_ell + " n_ell_L: " + n_ell_L + " n_ell_R: " + n_ell_R);
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

	private double calcLnTransRatioGrow(CGMBARTTreeNode T_i, CGMBARTTreeNode T_star, CGMBARTTreeNode grow_node) {
		int b = T_i.numLeaves();
		int p_adj = grow_node.pAdj();
		int n_adj = grow_node.nAdj();
		int w_2_star = T_star.numPruneNodesAvailable();
		int n_repeat = grow_node.splitValuesRepeated();
		return Math.log(b) + Math.log(p_adj) + Math.log(n_adj) - Math.log(w_2_star) - Math.log(n_repeat); 
	}

	protected boolean acceptProposal(double ln_u_0_1, double log_r){
		return ln_u_0_1 < log_r ? true : false;
	}

	/** The number of data points in a node that we can split on */
	protected static final int N_RULE = 5;	

	protected CGMBARTTreeNode pickGrowNode(CGMBARTTreeNode T) {
		ArrayList<CGMBARTTreeNode> growth_nodes = CGMBARTTreeNode.getTerminalNodesWithDataAboveOrEqualToN(T, N_RULE);
		
		//2 checks
		//a) If there is no nodes to grow, return null
		//b) If the node we picked CANNOT grow due to no available predictors, return null as well
		
		//do check a
		if (growth_nodes.size() == 0){
			System.out.println("no growth nodes in GROW step!");
			return null;
		}		
		
		//now we pick one of these nodes with enough data points randomly
		CGMBARTTreeNode growth_node = growth_nodes.get((int)Math.floor(StatToolbox.rand() * growth_nodes.size()));

		//do check b
		if (growth_node.pAdj() == 0){
			System.out.println("no attributes available in GROW step!");
			return null;			
		}
		//if we passed, we can use this node
		return growth_node;
	}
	
	protected Steps randomlyPickAmongTheTwoProposalSteps(CGMBARTTreeNode T) {
		double roll = StatToolbox.rand();
		if (roll < 0.5)
			return Steps.GROW;
		return Steps.PRUNE;	
	}
}
