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
public class CGMBARTPosteriorBuilder extends CGMPosteriorBuilder {

	private double hyper_sigsq_mu;
	private double sigsq;
	private double hyper_mu_bar;
	
	
	public CGMBARTPosteriorBuilder(CGMTreePriorBuilder tree_prior_builder) {
		super(tree_prior_builder);
	}
	
	public void setHyperparameters(double hyper_mu_bar, double hyper_sigsq_mu){
		this.hyper_mu_bar = hyper_mu_bar;
		this.hyper_sigsq_mu = hyper_sigsq_mu;
	}
	
//	private static final double ln_one_over_sqrt_two_pi = Math.log(1 / Math.sqrt(2 * Math.PI));
//	private static final double ln_two_pi = Math.log(2 * Math.PI);
	/**
	 * Note this is not the probability of Y given the tree, this is the probability
	 * of the r_i's that are in this tree, since each tree only fits the *remaining* data
	 */
	public double calculateLnProbYGivenTree(CGMTreeNode T) {
//		System.out.println("hyper_sigsq_mu: " + hyper_sigsq_mu);
		
		//first get the leaves
		ArrayList<CGMTreeNode> terminal_nodes = CGMTreeNode.getTerminalNodesWithDataAboveN(T, 0);
//		System.out.println("calculateLnProbYGivenTree num terminal_nodes: " + terminal_nodes.size());
		
//		int n = treePriorBuilder.getN();
		
		//calculate the constant term:
		double ln_prob = 0;//-n / 2 * ln_two_pi;
//		System.out.println("calculateLnProbYGivenTree after term 0: " + ln_prob);
		
		//now loop over all nodes and calculate term a
		
		for (CGMTreeNode node : terminal_nodes){
//			if (node.log_prop_lik == null){
				ln_prob += calculcateLnPropProbOfNode(node);
//			}
//			else {
//				ln_prob += node.log_prop_lik;
//			}
		}
//		System.out.println("calculateLnProbYGivenTree is prop to: " + ln_prob);
//		System.out.println("ProbYGivenTree :" + Math.pow(ln_prob, Math.E));
		
		//cache this value so we never have to calculate it again
		
		CGMBART.mh_iterations_full_record.print(
			TreeIllustration.two_digit_format.format(ln_prob) + ","
		);
		T.log_prop_lik = ln_prob;
		return ln_prob;
	}

	/**
	 * This calculates the marginalized ln(P(R_1, \ldots, R_N | \sigsq))
	 * by calculating g(\Rbar | \sigsq)
	 * 
	 * @param node
	 * @return
	 */
	private double calculcateLnPropProbOfNode(CGMTreeNode node) {
		double log_prob_node_ell = 0;
		log_prob_node_ell -= 0.5 * (Math.log(hyper_sigsq_mu) + Math.log(node.n / sigsq + 1));
		log_prob_node_ell += 
		
		DoubleMatrix rs = new DoubleMatrix(node.get_ys_in_data());
		DoubleMatrix mu = new DoubleMatrix(hyper_mu_bar, node.n);
		
		CGMBART.mh_iterations_full_record.print(
			TreeIllustration.two_digit_format.format(log_prob_node_ell) + ","
		);		
		//cache this for further use
		node.log_prop_lik = log_prob_node_ell;
		
		return log_prob_node_ell;
	}

	/** This grow step is different
	 * 
	 */
	protected Double createTreeProposalViaGrow(CGMTreeNode T) {
		
//		System.out.println("proposal via GROW  " + T.stringID());
		//find all terminals nodes that have **more** than N_RULE data
 
		ArrayList<CGMTreeNode> growth_nodes = CGMTreeNode.getTerminalNodesWithDataAboveN(T, N_RULE);
//		System.out.print("num growth nodes: " + growth_nodes.size() +":");
//		for (CGMTreeNode node : growth_nodes){
//			System.out.print(" " + node.stringID());
//		}
//		System.out.print("\n");
		//if there are no growth nodes at all, we need to get out with our skin intact,
		
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
