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


public class CGMRegressionMeanShiftPosteriorPlusEnhBuilder extends CGMRegressionMeanShiftPosteriorBuilder {

	public CGMRegressionMeanShiftPosteriorPlusEnhBuilder(CGMTreePriorBuilder tree_prior_builder, double[] y) {
		super(tree_prior_builder, y);
	}
	
	/**
	 * Take an internal node, then switch its rule
	 * 
	 * @param T_star	the tree to alter
	 */
	protected void createTreeProposalViaChange(CGMTreeNode T) {
		if (DEBUG_ITERATIONS){
			iteration_info.put("change_step", "CHANGE");
		}		
		System.out.println("proposal via CHANGE");
		//get all the internal nodes
		ArrayList<CGMTreeNode> internal_nodes = CGMTreeNode.findInternalNodes(T);
		if (internal_nodes.isEmpty()){
			System.out.println("no internal nodes");
			return;
		}
		
		ArrayList<CGMTreeNode> better_internal_nodes = new ArrayList<CGMTreeNode>();
		for (CGMTreeNode node : internal_nodes){
			int d = node.getGeneration();
			for (int i = 0; i < (int)Math.round(100 * 1 / (double)(d + 1)); i++){
				better_internal_nodes.add(node);
			}
		}
		
		//pick one internal node at random
		CGMTreeNode internal_node_to_change = better_internal_nodes.get(((int)Math.floor(StatToolbox.rand() * better_internal_nodes.size())));
		//now switch its rule
		internal_node_to_change.splitAttributeM = treePriorBuilder.assignSplitAttribute(internal_node_to_change);
		internal_node_to_change.splitValue = treePriorBuilder.assignSplitValue(internal_node_to_change.data, internal_node_to_change.splitAttributeM);
//		System.out.println("internal_node_to_change node: " + internal_node_to_change.stringID() + " rule: " + " X_" + internal_node_to_change.splitAttributeM + " < " + internal_node_to_change.splitValue);		
		if (DEBUG_ITERATIONS){
			iteration_info.put("changed_node", internal_node_to_change.stringID());
			iteration_info.put("split_attribute", internal_node_to_change.splitAttributeM + "");
			iteration_info.put("split_value", internal_node_to_change.splitValue + "");			
		}				
		//now we need to propagate this change all through its children and its children's children
		CGMTreeNode.propagateRuleChangeOrSwapThroughoutTree(internal_node_to_change, false);
	}

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
		System.out.println("proposal via GROW");
		ArrayList<CGMTreeNode> growth_nodes = CGMTreeNode.getTerminalNodesWithDataAboveOrEqualToN(T, N_RULE);
		System.out.print("num growth nodes: " + growth_nodes.size() +":");
//		for (CGMTreeNode node : growth_nodes){
//			System.out.print(" " + node.stringID());
//		}
//		System.out.print("\n");
		
		ArrayList<CGMTreeNode> better_growth_nodes = new ArrayList<CGMTreeNode>();
		for (CGMTreeNode node : growth_nodes){
			for (int i = 0; i < Math.sqrt(node.data.size()); i++){
				better_growth_nodes.add(node);
			}			
		}
		
		//if there are no growth nodes at all, we need to get out with our skin intact,
		//we return a probability of null
		if (growth_nodes.size() == 0){
			System.out.println("no growth nodes in GROW step!");
			return null;
		}
		//now we pick one of these nodes with enough data points randomly
		CGMTreeNode growth_node = better_growth_nodes.get((int)Math.floor(StatToolbox.rand() * better_growth_nodes.size()));
		//now we give it a split attribute and value and assign the children data
		treePriorBuilder.splitNodeAndAssignRule(growth_node);
		System.out.println("growth node: " + growth_node.stringID() + " rule: " + " X_" + growth_node.splitAttributeM + " < " + growth_node.splitValue);
		if (DEBUG_ITERATIONS){
			iteration_info.put("changed_node", growth_node.stringID());
			iteration_info.put("split_attribute", growth_node.splitAttributeM + "");
			iteration_info.put("split_value", growth_node.splitValue + "");
		}		
		//and now we need to return the probability that the growth node split
		return treePriorBuilder.calculateProbabilityOfSplitting(growth_node);
	}

}
