package CGM_BART;

import java.util.ArrayList;

import CGM_BART.CGMBART_07_mh.Steps;

public abstract class CGMBART_08_mh_with_change extends CGMBART_07_mh {
	private static final long serialVersionUID = -3874806337283953466L;
	
	 /** This does the MH
	 * 
	 * @param T_i				the original tree
	 * @param iteration_name	we just use this when naming the image file of this illustration
	 * @return 					the next tree (T_{i+1}) via one iteration of M-H
	 */
	protected CGMBARTTreeNode metroHastingsPosteriorTreeSpaceIteration(CGMBARTTreeNode T_i) {
//		System.out.println("iterateMHPosteriorTreeSpaceSearch");
		final CGMBARTTreeNode T_star = T_i.clone(true);
		//each proposal will calculate its own value, but this has to be initialized atop		
		double log_r = 0;
		switch (T_i.isStump() ? Steps.GROW : randomlyPickAmongTheProposalSteps(T_i)){
			case GROW:
				log_r = doMHGrowAndCalcLnR(T_i, T_star);
				break;
			case PRUNE:
				log_r = doMHPruneAndCalcLnR(T_i, T_star);
				break;
			case CHANGE:
				log_r = doMHChangeAndCalcLnR(T_i, T_star);
				break;				
		}		
		double ln_u_0_1 = Math.log(StatToolbox.rand());
//		if (log_r > Double.MIN_VALUE){
			System.out.println("u = " + Math.exp(ln_u_0_1) + 
					" <? r = " + 
					(Math.exp(log_r) < 0.00001 ? "damn small" : Math.exp(log_r)));		
//		}
		//ACCEPT/REJECT,STEP_name,log_prop_lik_o,log_prop_lik_f,log_r 
		CGMBART_03_debug.mh_iterations_full_record.println(
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

	protected double doMHChangeAndCalcLnR(CGMBARTTreeNode T_i, CGMBARTTreeNode T_star) {
		
		//first pick the change node on T_star, then find the corresponding node on T_I
		CGMBARTTreeNode changed_node = pickChangeNode(T_star);
		if (changed_node == null){
			System.err.println("proposal ln(r) = -oo DUE TO CANNOT CHANGE\n\n");
			return Double.NEGATIVE_INFINITY;
		}		
		
		
		CGMBARTTreeNode original_node = T_i.findCorrespondingNodeOnSimilarTree(changed_node);
		
		//now make the change
		changed_node.splitAttributeM = changed_node.pickRandomPredictorThatCanBeAssigned();
		changed_node.splitValue = changed_node.pickRandomSplitValue();
		changed_node.isLeaf = false;
		changed_node.left = new CGMBARTTreeNode(changed_node);
		changed_node.right = new CGMBARTTreeNode(changed_node);
		CGMBARTTreeNode.propagateDataByChangedRule(changed_node, true);
		
		//now calculate only the likelihood ratio of this MH step (since the tree structure and the transition ratios cancel out)
		double ln_likelihood_ratio_change = calcLnLikRatioChange(original_node, changed_node);
		
		System.out.println("CHANGE  <<" + original_node.stringLocation(true) 
			+ ">> ---- X_" + (original_node.splitAttributeM + 1) + 
			" < " + TreeIllustration.two_digit_format.format(original_node.splitValue)
			+ " ==> X_" + (changed_node.splitAttributeM + 1) + 
			" < " + TreeIllustration.two_digit_format.format(changed_node.splitValue) +		
			"\n  lik ratio: " + 
			(Math.exp(ln_likelihood_ratio_change) < 0.00001 ? "damn small" : Math.exp(ln_likelihood_ratio_change)));		
		return ln_likelihood_ratio_change;
	}
	
	protected CGMBARTTreeNode pickChangeNode(CGMBARTTreeNode T) {
		
		//2 checks
		//a) If this is the root, we can't prune so return null
		//b) If there are no prunable nodes (not sure how that could happen), return null as well
		
		if (T.isStump()){
			System.err.println("cannot change a stump!");
			return null;			
		}
		
		ArrayList<CGMBARTTreeNode> changeable_nodes = T.getPrunableAndChangeableNodes();
		if (changeable_nodes.size() == 0){
			System.err.println("no change nodes in CHANGE step!  T parent: " + T.parent + " T left: " + T.left + " T right: " + T.right);
			return null;
		}		
		
		//now we pick one of these nodes randomly
		return changeable_nodes.get((int)Math.floor(StatToolbox.rand() * changeable_nodes.size()));
	}	
	

	protected double calcLnLikRatioChange(CGMBARTTreeNode original_node, CGMBARTTreeNode changed_node) {
		double sigsq = gibbs_samples_of_sigsq.get(gibb_sample_num - 1);
		double sigsq_over_hyper_sigsq_mu = sigsq / hyper_sigsq_mu;
		int n_1_star = changed_node.left.n_eta;
		int n_2_star = changed_node.right.n_eta;
		int n_1 = original_node.left.n_eta;
		int n_2 = original_node.right.n_eta;
		
		double sum_R_1_star_i_s = changed_node.left.sumResponsesQuantitySqd();
		double sum_R_2_star_i_s = changed_node.right.sumResponsesQuantitySqd();
		double sum_R_1_i_s = original_node.left.sumResponsesQuantitySqd();
		double sum_R_2_i_s = original_node.right.sumResponsesQuantitySqd();
		
		double a = sum_R_1_star_i_s / (n_1_star + sigsq_over_hyper_sigsq_mu);
		double b = sum_R_2_star_i_s / (n_2_star + sigsq_over_hyper_sigsq_mu);
		double c = sum_R_1_i_s / (n_1 + sigsq_over_hyper_sigsq_mu);
		double d = sum_R_2_i_s / (n_2 + sigsq_over_hyper_sigsq_mu);
		
		System.out.println("calcLnLikRatioChange a: " + a + " b: " + b + " c: " + c + " d: " + d);
		return 1 / (2 * sigsq) * (a + b - c - d);
	}	
	
	
	//a hidden parameter in the BART model, P(PRUNE) = 1 - P(GROW) so only one needs to be defined here
	protected final static double PROB_GROW = 0.1;
	protected final static double PROB_CHANGE = 0.6;
	protected Steps randomlyPickAmongTheProposalSteps(CGMBARTTreeNode T) {
		double roll = StatToolbox.rand();
		if (roll < PROB_GROW)
			return Steps.GROW;
		else if (roll < PROB_CHANGE)
			return Steps.CHANGE;
		return Steps.PRUNE;	
	}
}
