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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import CGM_BART.CGMBART;
import CGM_BayesianCART1998.CGMPosteriorBuilder.Steps;
import CGM_Statistics.*;


public abstract class CGMPosteriorBuilder {

	/** we keep the most likely tree around */
	private CGMTreeNode most_likely_tree;
	/** we use the prior builder during out M-H iterations */
	protected CGMTreePriorBuilder treePriorBuilder;
	/** did the user press the "stop" button? */
	private boolean stop_bit;
	/** in order to keep highest log probability tree around, we keep */
	private double highest_log_probability;
	/** just some debug info about the current iteration in CSV format */
	protected HashMap<String, String> iteration_info;
	private int num_acceptances;


	public CGMPosteriorBuilder(CGMTreePriorBuilder treePriorBuilder){
		this.treePriorBuilder = treePriorBuilder;		
	}

	protected static final boolean DEBUG_ITERATIONS = true;
	private static PrintWriter mh_log_lik_iterations = null;
	private static PrintWriter mh_posterior_iterations = null;
	private static PrintWriter mh_num_leaves_iterations = null;
	private static PrintWriter mh_iterations_record = null;
	public static NumberFormat one_digit_format = NumberFormat.getInstance();
	static {
		one_digit_format.setMaximumFractionDigits(1);
	}
	static {
		try {
			if (DEBUG_ITERATIONS){
				mh_log_lik_iterations = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "mh_log_lik_iterations.txt")));
				mh_posterior_iterations = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "mh_posterior_iterations.txt")));
				mh_num_leaves_iterations = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "mh_num_leaves_iterations.txt")));
				mh_iterations_record = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "mh_iterations_record.txt")));
				TreeIllustration.DeletePreviousTreeIllustrations();
			}
		} catch (IOException e) {}
	}
	
	
	/**
	 * Build the posterior distribution
	 * 
	 * @param T_0					the initial tree's root (i.e. the tree generated by the prior)
	 * @param max_num_iterations 	the number of iterations to run
	 * @param i 
	 * @return						the best tree sampled from the posterior over all iterations
	 */
	public CGMTreeNode convergePosteriorAndFindMostLikelyTree(CGMTreeNode T_0, int max_num_iterations, int num_restart) {
		System.out.println("\nconvergePosterior max num iter: " + max_num_iterations + " num_restarts: " + num_restart);
		most_likely_tree = null;
		highest_log_probability = -Double.MAX_VALUE;
		num_acceptances = 0; //reset the number of acceptances
		int num_iterations = 0;
		iteration_info = new HashMap<String, String>();
		
		if (DEBUG_ITERATIONS){
			mh_log_lik_iterations.println(T_0.log_prop_lik);
			mh_posterior_iterations.println(T_0.log_prop_lik + Math.log(treePriorBuilder.probabilityOfTree(T_0))); //num_iter + "," + 
			mh_num_leaves_iterations.println(CGMTreeNode.numTerminalNodes(T_0));
			//print out how the tree looks
			iteration_info.put("num_restart", LeadingZeroes(num_restart, 3));
//			new TreeIllustration(T_0, iteration_info).WriteTitleAndSaveImage();
		}
		
		while (true){	
			num_iterations++;
			
			//record the info
			iteration_info = new HashMap<String, String>();
			iteration_info.put("num_restart", LeadingZeroes(num_restart, 3)); 
			iteration_info.put("num_iteration", LeadingZeroes(num_iterations, 5));
			System.out.println("num_restart: " + num_restart + " num_iteration: " + num_iterations);
			
			//iterate and write over previous tree			
			T_0 = iterateMHPosteriorTreeSpaceSearch(T_0, true);

			//do some debug info if necessary
			if (DEBUG_ITERATIONS){
				mh_log_lik_iterations.println(T_0.log_prop_lik);
				mh_posterior_iterations.println(T_0.log_prop_lik + Math.log(treePriorBuilder.probabilityOfTree(T_0))); //num_iter + "," + 
				mh_num_leaves_iterations.println(CGMTreeNode.numTerminalNodes(T_0));
				mh_iterations_record.println(iteration_info.get("num_restart") + "," + iteration_info.get("num_iterations"));				
			}			
			//bust out after we've hit maximum or if the user has pressed "stop"
			if (num_iterations == max_num_iterations || stop_bit){
				System.out.println("most likely tree: " + most_likely_tree);
				System.out.println("\n\ndone with " + max_num_iterations + " iterations final tree log-likelihood: " + calculateLnProbYGivenTree(most_likely_tree) + "\n\n");
				return most_likely_tree;
			}
		}
	}

	private static final String ZEROES = "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
	public static String LeadingZeroes(double num, int num_digits) {
		if (num < 10 && num_digits >= 2){
			return ZEROES.substring(0, num_digits - 1) + num;
		}
		else if (num < 100 && num_digits >= 3){
			return ZEROES.substring(0, num_digits - 2) + num;
		}
		else if (num < 1000 && num_digits >= 4){
			return ZEROES.substring(0, num_digits - 3) + num;
		}
		else if (num < 10000 && num_digits >= 5){
			return ZEROES.substring(0, num_digits - 4) + num;
		}
		else if (num < 100000 && num_digits >= 6){
			return ZEROES.substring(0, num_digits - 5) + num;
		}
		else if (num < 1000000 && num_digits >= 7){
			return ZEROES.substring(0, num_digits - 6) + num;
		}
		else if (num < 10000000 && num_digits >= 8){
			return ZEROES.substring(0, num_digits - 7) + num;
		}
		return String.valueOf(num);
	}
	public static String LeadingZeroes(int num, int num_digits) {
		if (num < 10 && num_digits >= 2){
			return ZEROES.substring(0, num_digits - 1) + num;
		}
		else if (num < 100 && num_digits >= 3){
			return ZEROES.substring(0, num_digits - 2) + num;
		}
		else if (num < 1000 && num_digits >= 4){
			return ZEROES.substring(0, num_digits - 3) + num;
		}
		else if (num < 10000 && num_digits >= 5){
			return ZEROES.substring(0, num_digits - 4) + num;
		}
		else if (num < 100000 && num_digits >= 6){
			return ZEROES.substring(0, num_digits - 5) + num;
		}
		else if (num < 1000000 && num_digits >= 7){
			return ZEROES.substring(0, num_digits - 6) + num;
		}
		else if (num < 10000000 && num_digits >= 8){
			return ZEROES.substring(0, num_digits - 7) + num;
		}
		return String.valueOf(num);
	}	

	/** a convenience to make the proposal steps look pretty */
	public enum Steps {GROW, PRUNE, CHANGE, SWAP};

	/**
	 * Iterates the Metropolis-Hastings algorithm by one step
	 * This is a public method because it gets called in the BART implementation
	 * 
	 * @param T_i				the original tree
	 * @param iteration_name	we just use this when naming the image file of this illustration
	 * @return 					the next tree (T_{i+1}) via one iteration of M-H
	 */
	public CGMTreeNode iterateMHPosteriorTreeSpaceSearch(CGMTreeNode T_i, boolean keep_highest_lik) {
		if (iteration_info == null){
			iteration_info = new HashMap<String, String>();
		}
		//make a copy, T_i will now serve as our ORIGINAL tree after the true original
		//gets modified via the proposal inducing function into T_star
		final CGMTreeNode T_star = T_i.clone(true);
		//each proposal will calculate its own value, but this has to be initialized atop		
		double log_r = 0;
		//now pick between GROW, PRUNE, CHANGE, SWAP
		switch (randomlyPickAmongTheFourProposalSteps()){
			case GROW:
				Double prob_split_grow = createTreeProposalViaGrow(T_star);
//				System.out.println("prob_split on grow: " + prob_split_grow);
//				System.out.println("num term nodes after growth: " + CGMTreeNode.numTerminalNodes(T_star));
				log_r = calculateLogRatioForGrow(T_i, prob_split_grow) + calculateLogRatioForChangeOrSwap(T_i, T_star);
				break;
			case PRUNE:
				Double prob_split_prune = createTreeProposalViaPrune(T_star);
//				System.out.println("prob_split on prune: " + prob_split_prune);
//				System.out.println("num term nodes after prune: " + CGMTreeNode.numTerminalNodes(T_star));
				log_r = calculateLogRatioForPrune(T_i, prob_split_prune) + calculateLogRatioForChangeOrSwap(T_i, T_star);
				break;
			case CHANGE:
				createTreeProposalViaChange(T_star);
				log_r = calculateLogRatioForChangeOrSwap(T_i, T_star);				
				break;				
			case SWAP:
				createTreeProposalViaSwap(T_star);
				log_r = calculateLogRatioForChangeOrSwap(T_i, T_star);				
				break;
		}
		
		//keep the most likely tree around
		if (keep_highest_lik){
//			System.out.println("log_r: " + log_r + " ln_prob_y_proposal: " + ln_prob_y_proposal + " highest_log_probability: " + highest_log_probability);
			if (T_star.log_prop_lik > highest_log_probability){			
				highest_log_probability = T_star.log_prop_lik;
				most_likely_tree = T_star.clone(true);
//				System.out.println("most likely tree log prob: " + ln_prob_y_proposal);
			}
		}
		double ln_u_0_1 = Math.log(Math.random());
		//ACCEPT/REJECT,STEP_name,log_prop_lik_o,log_prop_lik_f,log_r 
		CGMBART.mh_iterations_full_record.println(
			(acceptOrRejectProposal(ln_u_0_1, log_r) ? "A" : "R") + "," +  
			TreeIllustration.one_digit_format.format(log_r) + "," +
			TreeIllustration.two_digit_format.format(ln_u_0_1)
		);		
//		System.out.println("ln_u_0_1: " + ln_u_0_1 + " ln_r: " + log_r);
		if (acceptOrRejectProposal(ln_u_0_1, log_r)){ //accept proposal
			num_acceptances++;
			System.out.println("proposal ACCEPTED\n\n");
			if (DEBUG_ITERATIONS){
				iteration_info.put("num_acceptances", num_acceptances + "");
				iteration_info.put("ln_prob_y_proposal", T_star.log_prop_lik + "");
				iteration_info.put("log_r", one_digit_format.format(log_r));
				iteration_info.put("acc_or_rej", "ACCEPTED");
			}			
			return T_star;
		}
		//reject proposal
		System.out.println("proposal REJECTED\n\n");
		if (DEBUG_ITERATIONS){
			iteration_info.put("log_r", one_digit_format.format(log_r));
			iteration_info.put("acc_or_rej", "REJECTED");
		}
		return T_i;
	}
	
	protected boolean acceptOrRejectProposal(double ln_u_0_1, double log_r){
		return ln_u_0_1 < log_r ? true : false;
	}

	//can be overwritten to allow for more flexibility
	protected Steps randomlyPickAmongTheFourProposalSteps() {
		double roll = Math.random();
		if (roll < 0.25)
			return Steps.GROW;
		else if (roll < 0.5)
			return Steps.PRUNE;
		else if (roll < 0.75)
			return Steps.CHANGE;
		return Steps.SWAP;
	}

	/** The number of data points in a node that we can split on */
	protected static final int N_RULE = 5;
	
	/**
	 * See notes. We need to pick a random terminal node that has greater than N_RULE
	 * data points. Then we need to grow it by choosing a split rule
	 * 
	 * @param T		the tree that will become the proposal tree
	 * @return		the probability of splitting at the node that is grown
	 */
	protected Double createTreeProposalViaGrow(CGMTreeNode T) {
		//find all terminals nodes that have **more** than N_RULE data
		if (DEBUG_ITERATIONS){
			iteration_info.put("change_step", "GROW");
		}
//		System.out.println("proposal via GROW  " + T.stringID());
		ArrayList<CGMTreeNode> growth_nodes = CGMTreeNode.getTerminalNodesWithDataAboveN(T, N_RULE);
//		System.out.print("num growth nodes: " + growth_nodes.size() +":");
//		for (CGMTreeNode node : growth_nodes){
//			System.out.print(" " + node.stringID());
//		}
//		System.out.print("\n");
		//if there are no growth nodes at all, we need to get out with our skin intact,
		//we return a probability of null
		if (growth_nodes.size() == 0){
//			System.out.println("no growth nodes in GROW step!");
			return null;
		}
		//now we pick one of these nodes with enough data points randomly
		CGMTreeNode growth_node = growth_nodes.get((int)Math.floor(Math.random() * growth_nodes.size()));
		//now we give it a split attribute and value and assign the children data
		treePriorBuilder.splitNodeAndAssignRule(growth_node);
//		System.out.println("growth node: " + growth_node.stringID() + " rule: " + " X_" + growth_node.splitAttributeM + " < " + growth_node.splitValue);
		if (DEBUG_ITERATIONS){
			iteration_info.put("changed_node", growth_node.stringID());
			iteration_info.put("split_attribute", growth_node.splitAttributeM + "");
			iteration_info.put("split_value", growth_node.splitValue + "");
		}		
		//and now we need to return the probability that the growth node split
		return treePriorBuilder.calculateProbabilityOfSplitting(growth_node);
	}
	

	/**
	 * See notes. We need to calculate the ln(r) value which has 5 terms:
	 * + \natlog{\probsub{SPLIT}{\cdot}} 
	 * + \natlog{b} 
	 * - \natlog{2b^*} 
	 * + \natlog{\prob{\y | T_*, \X}} 
	 * - \natlog{\prob{\y | T_i, \X}}
	 * 
	 * @param T_i			the original tree
	 * @param T_star		the proposal tree
	 * @param probSplit 	the probability that this tree was grown at that particular place where we grew it
	 * @return				the ln(r) calculation
	 */
		//if we didn't split at all, return 0 so tree doesn't pass
	private double calculateLogRatioForGrow(CGMTreeNode T_i, Double prob_split) {
		if (prob_split == null){
//			System.out.println("probSplit null in calculateLogRatioForGrow");
			return 0;
		}
		//now we calculate each quantity
		//first calculate the log of the number of terminal nodes
		//in the original tree
		double ln_b = Math.log(CGMTreeNode.numTerminalNodes(T_i));
		//now calculate two times the log of the number of terminal
		//nodes that satisfy the minimum data requirement in the original tree
		double lnbtwobstar = Math.log(2 * CGMTreeNode.numTerminalNodesDataAboveN(T_i, N_RULE));
//		System.out.println("calculateLogRatioForGrow p(y|T*) / p(y|Ti) = " + Math.pow(Math.E, ln_prob_y_proposal - ln_prob_y_original));
		//now add everything up appropriately
		return Math.log(prob_split) + ln_b - lnbtwobstar;
	}

	/**
	 * See notes. We need to pick a random "branch" node (i.e. a node whose
	 * children are both leaves), then prune the tree at that point by turning it into a leaf.
	 * 
	 * @param T		the tree that will become the proposal tree
	 * @return		the probability of splitting at the node that is pruned
	 */	
	protected Double createTreeProposalViaPrune(CGMTreeNode T) {
		if (DEBUG_ITERATIONS){
			iteration_info.put("change_step", "PRUNE");
		}
		//get a random terminal node (all terminal nodes accepted)
//		System.out.println("proposal via PRUNE  " + T.stringID());
		ArrayList<CGMTreeNode> terminal_nodes = CGMTreeNode.getTerminalNodesWithDataAboveN(T, 0);
//		System.out.print("num terminal nodes: " + terminal_nodes.size());
//		for (CGMTreeNode node : terminal_nodes){
//			System.out.print(" " + node.stringID());
//		}	
//		System.out.print("\n");	
		//now select out the terminal node's parents whose both children are leaves, call them "branch nodes"
		HashSet<CGMTreeNode> branch_nodes = CGMTreeNode.selectBranchNodesWithTwoLeaves(terminal_nodes);
//		System.out.print("num branch nodes: " + branch_nodes.size() + ":");
//		for (CGMTreeNode node : branch_nodes){
//			System.out.print(" " + node.stringID());
//		}
//		System.out.print("\n");		
		//if there are no branch nodes at all, we need to get out with our skin intact,
		//we return a probability of null
		if (branch_nodes.size() == 0){
//			System.out.println("no branch nodes in PRUNE step!");
			return null;
		}		
		//now we pick one of these nodes randomly AND take its parent
		//this involves some retarded Java gymnastics
		CGMTreeNode prune_node = ((CGMTreeNode)branch_nodes.toArray()[((int)Math.floor(Math.random() * branch_nodes.size()))]);
		//if the prune node happened to be the root node,
		//its parent is null, and we have to jet big time
		if (prune_node == null){
//			System.out.println("no prune nodes in PRUNE step!");
			return null;			
		}
//		System.out.println("prune_node node: " + prune_node.stringID() + " rule: " + " X_" + prune_node.splitAttributeM + " < " + prune_node.splitValue);		
		//now we prune this node
		CGMTreeNode.pruneTreeAt(prune_node);
		if (DEBUG_ITERATIONS){
			iteration_info.put("changed_node", prune_node.stringID());
		}		
		//and now we need to return the probability that the prune node split
		return treePriorBuilder.calculateProbabilityOfSplitting(prune_node);
	}

	/**
	 * See notes. We need to calculate the ln(r) value which has 5 terms:
	 * - \natlog{\probsub{SPLIT}{\cdot}} 
	 * + \natlog{b^*} 
	 * - \natlog{2b} 
	 * + \natlog{\prob{\y | T_*, \X}} 
	 * - \natlog{\prob{\y | T_i, \X}}
	 * 
	 * @param T_i			the original tree
	 * @param T_star		the proposal tree
	 * @param probSplit 	the probability that this tree was grown at that particular place where it was pruned
	 * @return				the ln(r) calculation
	 */
	private double calculateLogRatioForPrune(CGMTreeNode T_i, Double prob_split) {
		//if we didn't prune at all, return 0 so tree doesn't pass
		if (prob_split == null){
//			System.out.println("probSplit null in calculateLogRatioForPrune");
			return 0;
		}
		//now we calculate each quantity
		//first calculate the log of the number of terminal nodes
		//in the original tree
		double ln_bstar = Math.log(CGMTreeNode.numTerminalNodesDataAboveN(T_i, N_RULE));
		//now calculate two times the log of the number of terminal
		//nodes that satisfy the minimum data requirement in the original tree
		double ln_twob = Math.log(2 * CGMTreeNode.numTerminalNodes(T_i));
//		System.out.println("calculateLogRatioForPrune p(y|T*) / p(y|Ti) = " + Math.pow(Math.E, ln_prob_y_proposal - ln_prob_y_original));
		//now add everything up appropriately
		return -Math.log(prob_split) + ln_bstar - ln_twob;
	}
	
	/**
	 * Calculate the log probability of y given the tree T
	 * 
	 * @param T		The tree
	 * @return		the log probability
	 */
	protected abstract double calculateLnProbYGivenTree(CGMTreeNode T);

	private double calculateLogRatioForChangeOrSwap(CGMTreeNode T_i, CGMTreeNode T_star) {
		double ln_prob_y_proposal = calculateLnProbYGivenTree(T_star);
		double ln_prob_y_original = calculateLnProbYGivenTree(T_i);
//		System.out.println("prop ln(P(R|T*)) = " + ln_prob_y_proposal);
//		System.out.println("prop ln(P(R|Ti)) = " + ln_prob_y_original);
//		System.out.println("calculateLogRatioForChangeOrSwap p(y|T*) / p(y|Ti) = " + Math.pow(Math.E, ln_prob_y_proposal - ln_prob_y_original));
		return ln_prob_y_proposal - ln_prob_y_original;
	}
	
	protected CGMTreeNode pickChangeNode(ArrayList<CGMTreeNode> internal_nodes) {
		//return a random one
		return internal_nodes.get(((int)Math.floor(Math.random() * internal_nodes.size())));
	}

	/**
	 * Take an internal node, then switch its rule
	 * 
	 * @param T_star	the tree to alter
	 */
	protected void createTreeProposalViaChange(CGMTreeNode T) {
		System.out.println("proposal via CHANGE  " + T.stringID());
		
		if (DEBUG_ITERATIONS){
			iteration_info.put("change_step", "CHANGE");
		}		

		//pick one internal node at random
		ArrayList<CGMTreeNode> internal_nodes = CGMTreeNode.findInternalNodes(T);
		if (internal_nodes.isEmpty()){
			System.out.println("no internal nodes");
			return;
		}		
		CGMTreeNode internal_node_to_change = pickChangeNode(internal_nodes); 

		//now switch its rule
		Integer prevsplitAttributeM = internal_node_to_change.splitAttributeM;
		Double presplitValue = internal_node_to_change.splitValue;
		internal_node_to_change.splitAttributeM = treePriorBuilder.assignSplitAttribute(internal_node_to_change);
		internal_node_to_change.splitValue = treePriorBuilder.assignSplitValue(internal_node_to_change.data, internal_node_to_change.splitAttributeM);
		internal_node_to_change.log_prop_lik = null; //release the cache
		System.out.println("internal_node_to_change node: " + internal_node_to_change.stringID() + " rule: " + " X_" + internal_node_to_change.splitAttributeM + " < " + internal_node_to_change.splitValue + "   total num int nodes: " + internal_nodes.size());		
		if (DEBUG_ITERATIONS){
			iteration_info.put("changed_node", internal_node_to_change.stringID());
			iteration_info.put("split_attribute", internal_node_to_change.splitAttributeM + "");
			iteration_info.put("split_value", internal_node_to_change.splitValue + "");	
			CGMBART.mh_iterations_full_record.print(
				"CHANGE" + "," + 
				internal_node_to_change.stringID() + "," + 
				internal_node_to_change.stringLocation(true) + "," +
				"X_" + (prevsplitAttributeM + 1) + "," + 
				LeadingZeroes(Double.parseDouble(TreeIllustration.one_digit_format.format(presplitValue)), 4) + "," +					
				"X_" + (internal_node_to_change.splitAttributeM + 1) + "," + 
				LeadingZeroes(Double.parseDouble(TreeIllustration.one_digit_format.format(internal_node_to_change.splitValue)), 4) + ","		
			);
		}				
		//now we need to propagate this change all through its children and its children's children
		CGMTreeNode.propagateRuleChangeOrSwapThroughoutTree(internal_node_to_change, false);
	}
	
	protected void createTreeProposalViaSwap(CGMTreeNode T) {
		if (DEBUG_ITERATIONS){
			iteration_info.put("change_step", "SWAP");
		}			
//		System.out.println("proposal via SWAP  " + T.stringID());
		//get all the doubly internal nodes (i.e. internal nodes whose children are internal nodes as well)
		ArrayList<CGMTreeNode> doubly_internal_nodes = CGMTreeNode.findDoublyInternalNodes(T);
		if (doubly_internal_nodes.isEmpty()){
//			System.out.println("no doubly internal nodes");
			return;
		}
		//pick one internal node at random
		CGMTreeNode doubly_internal_node_to_swap = doubly_internal_nodes.get(((int)Math.floor(Math.random() * doubly_internal_nodes.size())));
		//now get both its children
		CGMTreeNode left_child = doubly_internal_node_to_swap.left;
		CGMTreeNode right_child = doubly_internal_node_to_swap.right;
		//release the cache
		doubly_internal_node_to_swap.log_prop_lik = null;
		left_child.log_prop_lik = null;
		right_child.log_prop_lik = null;
		
		Integer prevsplitAttributeM = doubly_internal_node_to_swap.splitAttributeM;
		Double presplitValue = doubly_internal_node_to_swap.splitValue;		
		//in the off chance that both children have the same rules...
		if (left_child.splitAttributeM == right_child.splitAttributeM && left_child.splitValue == right_child.splitValue){
			//swap the parent's rule with the childrens' rule
			Integer save_parent_attribute = doubly_internal_node_to_swap.splitAttributeM;
			Double save_parent_value = doubly_internal_node_to_swap.splitValue;
			doubly_internal_node_to_swap.splitAttributeM = left_child.splitAttributeM;
			doubly_internal_node_to_swap.splitValue = left_child.splitValue;
			left_child.splitAttributeM = save_parent_attribute;
			left_child.splitValue = save_parent_value;
			right_child.splitAttributeM = save_parent_attribute;
			right_child.splitValue = save_parent_value;
		}
		else {
			//pick one of the children at random
			CGMTreeNode child = Math.random() < 0.5 ? right_child : left_child;
			//swap the parent's rule with the children's rule
			Integer save_parent_attribute = doubly_internal_node_to_swap.splitAttributeM;
			Double save_parent_value = doubly_internal_node_to_swap.splitValue;
			doubly_internal_node_to_swap.splitAttributeM = child.splitAttributeM;
			doubly_internal_node_to_swap.splitValue = child.splitValue;	
			child.splitAttributeM = save_parent_attribute;
			child.splitValue = save_parent_value;
		}
//		System.out.println("doubly_internal_node_to_swap node: " + doubly_internal_node_to_swap.stringID() + " rule: " + " X_" + doubly_internal_node_to_swap.splitAttributeM + " < " + doubly_internal_node_to_swap.splitValue);		
		if (DEBUG_ITERATIONS){
			iteration_info.put("changed_node", doubly_internal_node_to_swap.stringID());
			iteration_info.put("split_attribute", doubly_internal_node_to_swap.splitAttributeM + "");
			iteration_info.put("split_value", doubly_internal_node_to_swap.splitValue + "");
			CGMBART.mh_iterations_full_record.print(
				"SWAP" + "," + 
				doubly_internal_node_to_swap.stringID() + "," + 
				doubly_internal_node_to_swap.stringLocation(true) + "," +
				"X_" + (prevsplitAttributeM + 1) + " < " + 
				TreeIllustration.one_digit_format.format(presplitValue) + "," +					
				"X_" + (doubly_internal_node_to_swap.splitAttributeM + 1) + " < " + 
				TreeIllustration.one_digit_format.format(doubly_internal_node_to_swap.splitValue) + ","		
			);			
		}			
		//now we need to propagate this change all through its children and its children's children
		CGMTreeNode.propagateRuleChangeOrSwapThroughoutTree(doubly_internal_node_to_swap, false);
	}

	public void StopBuilding() {
		stop_bit = true;
	}


	public void close_debug_information() {
		if (DEBUG_ITERATIONS){
			mh_log_lik_iterations.close();
			mh_posterior_iterations.close();
			mh_num_leaves_iterations.close();
			mh_iterations_record.close();			
		}
	}	

}