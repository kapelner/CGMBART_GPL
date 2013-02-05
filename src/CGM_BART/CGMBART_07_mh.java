package CGM_BART;

import java.io.Serializable;
import java.util.ArrayList;


public abstract class CGMBART_07_mh extends CGMBART_06_gibbs_internal implements Serializable {
	private static final long serialVersionUID = 1825856510284398699L;
	private static final boolean DEBUG_MH = false;


	//this enum has to include all potential types of steps, even if they aren't used in this class's implementation
	public enum Steps {GROW, PRUNE, CHANGE};
	/**
	 * Iterates the Metropolis-Hastings algorithm by one step
	 * This does the MH
	 * 
	 * @param T_i				the original tree
	 * @param iteration_name	we just use this when naming the image file of this illustration
	 * @return 					the next tree (T_{i+1}) via one iteration of M-H
	 */
	@SuppressWarnings("incomplete-switch") //not all steps are used in this class's implementation
	protected CGMBARTTreeNode metroHastingsPosteriorTreeSpaceIteration(CGMBARTTreeNode T_i) {
//		System.out.println("iterateMHPosteriorTreeSpaceSearch");
		final CGMBARTTreeNode T_star = T_i.clone();
//		System.out.println("SampleTree T_i \n responses: " + 
//				Tools.StringJoin(T_i.responses) + 
//				"\n indicies " + Tools.StringJoin(T_i.indicies));		
//		System.out.println("SampleTree T_star \n responses: " + 
//				Tools.StringJoin(T_star.responses) + 
//				"\n indicies " + Tools.StringJoin(T_star.indicies)); 		
		//each proposal will calculate its own value, but this has to be initialized atop		
		double log_r = 0;
		
		//if it's a stump force a GROW change, otherwise pick randomly according to the "hidden parameters"
		switch (T_i.isStump() ? Steps.GROW : randomlyPickAmongTheProposalSteps(T_i)){
			case GROW:
				log_r = doMHGrowAndCalcLnR(T_i, T_star);
				break;
			case PRUNE:
				log_r = doMHPruneAndCalcLnR(T_i, T_star);
				break;
		}		
		double ln_u_0_1 = Math.log(StatToolbox.rand());
//		if (log_r > Double.MIN_VALUE){
		if (DEBUG_MH){
			System.out.println("ln u = " + ln_u_0_1 + 
					" <? ln(r) = " + 
					(log_r < -99999 ? "damn small" : log_r));
		}
//		}
		//ACCEPT/REJECT,STEP_name,log_prop_lik_o,log_prop_lik_f,log_r 
//		CGMBART_03_debug.mh_iterations_full_record.println(
//			(acceptProposal(ln_u_0_1, log_r) ? "A" : "R") + "," +  
//			TreeIllustration.one_digit_format.format(log_r) + "," +
//			TreeIllustration.two_digit_format.format(ln_u_0_1)
//		);		
//		System.out.println("ln_u_0_1: " + ln_u_0_1 + " ln_r: " + log_r);
		if (acceptProposal(ln_u_0_1, log_r)){ //accept proposal
			if (DEBUG_MH){
				System.out.println("proposal ACCEPTED\n\n");
			}
			return T_star;
		}		
		//reject proposal
		if (DEBUG_MH){
			System.out.println("proposal REJECTED\n\n");
		}
		return T_i;
	}

	protected double doMHGrowAndCalcLnR(CGMBARTTreeNode T_i, CGMBARTTreeNode T_star) {
//		System.out.println("doMHGrowAndCalcLnR on " + T_star.stringID() + " old depth: " + T_star.deepestNode());
		CGMBARTTreeNode grow_node = pickGrowNode(T_star);
		//if we can't grow, reject offhand
		if (grow_node == null){ // || grow_node.generation >= 2
//			System.out.println("no valid grow nodes    proposal ln(r) = -oo DUE TO CANNOT GROW\n\n");
			return Double.NEGATIVE_INFINITY;					
		}
		
		//now start the growth process
		//first pick the attribute and then the split
		grow_node.splitAttributeM = pickRandomPredictorThatCanBeAssigned(grow_node);
		grow_node.splitValue = grow_node.pickRandomSplitValue();
//		System.out.print("split_value = " + split_value);
		//inform the user if things go awry
		if (grow_node.splitValue == CGMBARTTreeNode.BAD_FLAG_double){
			System.out.print("ERROR!!! GROW <<" + grow_node.stringLocation(true) + ">> ---- X_" + (grow_node.splitAttributeM + 1) + "  proposal ln(r) = -oo DUE TO NO SPLIT VALUES\n\n");
			return Double.NEGATIVE_INFINITY;					
		}			
		grow_node.isLeaf = false;
		grow_node.left = new CGMBARTTreeNode(grow_node);
		grow_node.right = new CGMBARTTreeNode(grow_node);	
		grow_node.propagateDataByChangedRule();

		if (grow_node.left.n_eta <= N_RULE || grow_node.right.n_eta <= N_RULE){
			if (DEBUG_MH){
				System.out.println("ERR: cannot split a node where daughter only has NO data points   proposal ln(r) = -oo DUE TO CANNOT GROW\n\n");
			}
			return Double.NEGATIVE_INFINITY;
		}
//		System.out.print("grow_node.splitValue = " + grow_node.splitValue);
		
//		System.out.println("grow_node: " + grow_node.stringID() + " new depth: " + T_star.deepestNode() + " " + grow_node.deepestNode());
		double ln_transition_ratio_grow = calcLnTransRatioGrow(T_i, T_star, grow_node);
		double ln_likelihood_ratio_grow = calcLnLikRatioGrow(grow_node);
		double ln_tree_structure_ratio_grow = calcLnTreeStructureRatioGrow(grow_node);
		
		if (DEBUG_MH){
			System.out.println("GROW  <<" + grow_node.stringLocation(true) + ">> ---- X_" + (grow_node.splitAttributeM + 1) + 
				" < " + TreeIllustration.two_digit_format.format(grow_node.splitValue) + 
				"\n  ln trans ratio: " + ln_transition_ratio_grow + " ln lik ratio: " + ln_likelihood_ratio_grow + " ln structure ratio: " + ln_tree_structure_ratio_grow +			
				"\n  trans ratio: " + 
				(Math.exp(ln_transition_ratio_grow) < 0.00001 ? "damn small" : Math.exp(ln_transition_ratio_grow)) +
				"  lik ratio: " + 
				(Math.exp(ln_likelihood_ratio_grow) < 0.00001 ? "damn small" : Math.exp(ln_likelihood_ratio_grow)) +
				"  structure ratio: " + 
				(Math.exp(ln_tree_structure_ratio_grow) < 0.00001 ? "damn small" : Math.exp(ln_tree_structure_ratio_grow)));
		}
		return ln_transition_ratio_grow + ln_likelihood_ratio_grow + ln_tree_structure_ratio_grow;
	}
	
	protected double doMHPruneAndCalcLnR(CGMBARTTreeNode T_i, CGMBARTTreeNode T_star) {
		CGMBARTTreeNode prune_node = pickPruneNode(T_star);
		//if we can't grow, reject offhand
		if (prune_node == null){
			System.err.println("proposal ln(r) = -oo DUE TO CANNOT PRUNE\n\n");
			return Double.NEGATIVE_INFINITY;
		}				
		double ln_transition_ratio_prune = calcLnTransRatioPrune(T_i, T_star, prune_node);
		double ln_likelihood_ratio_prune = -calcLnLikRatioGrow(prune_node); //inverse of before (will speed up later)
		double ln_tree_structure_ratio_prune = -calcLnTreeStructureRatioGrow(prune_node);
		
		if (DEBUG_MH){
			System.out.println("PRUNE <<" + prune_node.stringLocation(true) + 
					">> ---- X_" + (prune_node.splitAttributeM == CGMBARTTreeNode.BAD_FLAG_int ? "null" : (prune_node.splitAttributeM + 1)) + " < " + TreeIllustration.two_digit_format.format(prune_node.splitValue == CGMBARTTreeNode.BAD_FLAG_double ? Double.NaN : prune_node.splitValue) + 
				"\n  ln trans ratio: " + ln_transition_ratio_prune + " ln lik ratio: " + ln_likelihood_ratio_prune + " ln structure ratio: " + ln_tree_structure_ratio_prune +
				"\n  trans ratio: " + 
				(Math.exp(ln_transition_ratio_prune) < 0.00001 ? "damn small" : Math.exp(ln_transition_ratio_prune)) +
				"  lik ratio: " + 
				(Math.exp(ln_likelihood_ratio_prune) < 0.00001 ? "damn small" : Math.exp(ln_likelihood_ratio_prune)) +
				"  structure ratio: " + 
				(Math.exp(ln_tree_structure_ratio_prune) < 0.00001 ? "damn small" : Math.exp(ln_tree_structure_ratio_prune)));
		}
		CGMBARTTreeNode.pruneTreeAt(prune_node);
		return ln_transition_ratio_prune + ln_likelihood_ratio_prune + ln_tree_structure_ratio_prune;
	}	
	
	protected CGMBARTTreeNode pickPruneNode(CGMBARTTreeNode T) {
		
		//2 checks
		//a) If this is the root, we can't prune so return null
		//b) If there are no prunable nodes (not sure how that could happen), return null as well
		
		if (T.isStump()){
			System.err.println("cannot prune a stump!");
			return null;			
		}
		
		ArrayList<CGMBARTTreeNode> prunable_nodes = T.getPrunableAndChangeableNodes();
		if (prunable_nodes.size() == 0){
			System.err.println("no prune nodes in PRUNE step!  T parent: " + T.parent + " T left: " + T.left + " T right: " + T.right);
			return null;
		}		
		
		//now we pick one of these nodes randomly
		return prunable_nodes.get((int)Math.floor(StatToolbox.rand() * prunable_nodes.size()));
	}
	
	protected double calcLnTransRatioPrune(CGMBARTTreeNode T_i, CGMBARTTreeNode T_star, CGMBARTTreeNode prune_node) {
		int w_2 = T_i.numPruneNodesAvailable();
		int b = T_i.numLeaves();
		double p_adj = pAdj(prune_node);
		int n_adj = prune_node.nAdj();
		return Math.log(w_2) - Math.log(b - 1) - Math.log(p_adj) - Math.log(n_adj); 
	}

	protected double calcLnTreeStructureRatioGrow(CGMBARTTreeNode grow_node) {
		int d_eta = grow_node.depth;
		double p_adj = pAdj(grow_node);
		int n_adj = grow_node.nAdj();
//		System.out.println("calcLnTreeStructureRatioGrow d_eta: " + d_eta + " p_adj: " + p_adj + " n_adj: " + n_adj + " n_repeat: " + n_repeat + " ALPHA: " + ALPHA + " BETA: " + BETA);
		return Math.log(ALPHA) 
				+ 2 * Math.log(1 - ALPHA / Math.pow(2 + d_eta, BETA))
				- Math.log(Math.pow(1 + d_eta, BETA) - ALPHA)
				- Math.log(p_adj) 
				- Math.log(n_adj);
	}	

	/** The number of data points in a node that we can split on */
	protected static int N_RULE = 0;	

	protected double calcLnLikRatioGrow(CGMBARTTreeNode grow_node) {
		double sigsq = gibbs_samples_of_sigsq[gibbs_sample_num - 1];
		int n_ell = grow_node.n_eta;
		int n_ell_L = grow_node.left.n_eta;
		int n_ell_R = grow_node.right.n_eta;
		
//		System.out.println(" sigsq: " + sigsq);
//		System.out.println("calcLnLikRatioGrow n_ell: " + n_ell + " n_ell_L: " + n_ell_L + " n_ell_R: " + n_ell_R);

		//now go ahead and calculate it out	in an organized fashion:
		double sigsq_plus_n_ell_hyper_sisgsq_mu = sigsq + n_ell * hyper_sigsq_mu;
		double sigsq_plus_n_ell_L_hyper_sisgsq_mu = sigsq + n_ell_L * hyper_sigsq_mu;
		double sigsq_plus_n_ell_R_hyper_sisgsq_mu = sigsq + n_ell_R * hyper_sigsq_mu;
		double c = 0.5 * (
				Math.log(sigsq) 
				+ Math.log(sigsq_plus_n_ell_hyper_sisgsq_mu) 
				- Math.log(sigsq_plus_n_ell_L_hyper_sisgsq_mu) 
				- Math.log(sigsq_plus_n_ell_R_hyper_sisgsq_mu));
		double d = hyper_sigsq_mu / (2 * sigsq);
		double e = grow_node.left.sumResponsesQuantitySqd() / sigsq_plus_n_ell_L_hyper_sisgsq_mu
				+ grow_node.right.sumResponsesQuantitySqd() / sigsq_plus_n_ell_R_hyper_sisgsq_mu
				- grow_node.sumResponsesQuantitySqd() / sigsq_plus_n_ell_hyper_sisgsq_mu;
//		System.out.println("calcLnLikRatioGrow c: " + c + " d: " + d + " e: " + e);
		return c + d * e;
	}

	protected double calcLnTransRatioGrow(CGMBARTTreeNode T_i, CGMBARTTreeNode T_star, CGMBARTTreeNode node_grown_in_Tstar) {
		int b = T_i.numLeaves();
		double p_adj = pAdj(node_grown_in_Tstar);
		int n_adj = node_grown_in_Tstar.nAdj();
		int w_2_star = T_star.numPruneNodesAvailable();
//		System.out.println("calcLnTransRatioGrow b:" + b + " p_adj: " + p_adj + " n_adj: " + n_adj + " w_2_star:" + w_2_star + " n_repeat:" + n_repeat);
		return Math.log(b) + Math.log(p_adj) + Math.log(n_adj) - Math.log(w_2_star); 
	}

	protected boolean acceptProposal(double ln_u_0_1, double log_r){
		return ln_u_0_1 < log_r ? true : false;
	}

	protected CGMBARTTreeNode pickGrowNode(CGMBARTTreeNode T) {
		ArrayList<CGMBARTTreeNode> growth_nodes = T.getTerminalNodesWithDataAboveOrEqualToN(2);
		
		//2 checks
		//a) If there is no nodes to grow, return null
		//b) If the node we picked CANNOT grow due to no available predictors, return null as well
		
		//do check a
		if (growth_nodes.size() == 0){
			System.err.println("no growth nodes in GROW step!");
			return null;
		}		
		
		//now we pick one of these nodes with enough data points randomly
		CGMBARTTreeNode growth_node = growth_nodes.get((int)Math.floor(StatToolbox.rand() * growth_nodes.size()));

		//do check b
		if (pAdj(growth_node) == 0){
			System.err.println("no attributes available in GROW step!");
			return null;			
		}
		//if we passed, we can use this node
		return growth_node;
	}
	
	//a hidden parameter in the BART model, P(PRUNE) = 1 - P(GROW) so only one needs to be defined here
	protected final static double PROB_GROW = 0.5;
	protected Steps randomlyPickAmongTheProposalSteps(CGMBARTTreeNode T) {
		double roll = StatToolbox.rand();
		if (roll < PROB_GROW)
			return Steps.GROW;
		return Steps.PRUNE;	
	}
}
