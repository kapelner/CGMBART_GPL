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

package CGM_Statistics;

import java.util.ArrayList;
import java.util.List;

public abstract class CGMTreePriorBuilder {
	
	protected ArrayList<double[]> X_y;
	private int p;	
	



	public CGMTreePriorBuilder(ArrayList<double[]> X_y, int p){
		this.X_y = X_y;
		this.p = p;
	}
	
	public abstract double getAlpha();
	public abstract double getBeta();
	
	public int getP() {
		return p;
	}	
	
	public int getN() {
		return X_y.size();
	}		


	public CGMTreeNode buildTreeStructureBasedOnPrior() {
		//we're going to build this exactly like CGM 938-9
		//first build the root (by definition the root is parentless)
		CGMTreeNode root = new CGMTreeNode(null, X_y);
		//now determine recursively the size and shape of tree
		constructSizeAndShape(root);
		//return the tree by sending back the root
		return root;
	}
	
	///// all functions related to splitting and node placement prior ($\T$)


	private void constructSizeAndShape(CGMTreeNode node) {
		//do we need to do this in the posterior
		double prob_split = calculateProbabilityOfSplitting(node);
		
//		System.out.println("node: " + node.stringID() + " parent: " + (node.parent == null ? "null" : node.parent.stringID()));
		//actually flip the Bernoulli coin
		if (Math.random() < prob_split){ 
			//if we choose to split upon rolling the dice...
//			System.out.println("SPLIT on p = " + prob_split);
			//and if we actually do split, then grow the children
			if (splitNodeAndAssignRule(node)){
				constructSizeAndShape(node.left);
				constructSizeAndShape(node.right);				
			}
		}
		else {
//			System.out.println("FAILED TO SPLIT on p = " + prob_split);
			//we choose to NOT split upon rolling the dice...
			markNodeAsLeaf(node);
//			CGMTreeNode.DebugNode(node);
		}
	}

	public boolean splitNodeAndAssignRule(CGMTreeNode node) {		
		//first assign a split attribute
		node.splitAttributeM = assignSplitAttribute(node);
		
		//if 
		//a) we don't have any attributes left to split on, 
		//b) there's only one data point left
		//then this node automatically becomes a leaf, otherwise split it
		if (node.splitAttributeM == null || node.data.size() == 1){
			markNodeAsLeaf(node);
			return false; //we did not do a split
		}	
		else {
			//we're no longer a leaf if we once were
			node.isLeaf = false;
			node.klass = null;
			//assign a splitting value
			node.splitValue = assignSplitValue(node.data, node.splitAttributeM);			
			//split the data correctly
			ClassificationAndRegressionTree.SortAtAttribute(node.data, node.splitAttributeM);
			int n_split = ClassificationAndRegressionTree.getSplitPoint(node.data, node.splitAttributeM, node.splitValue);
			//now build the node offspring
			node.left = new CGMTreeNode(node, ClassificationAndRegressionTree.getLowerPortion(node.data, n_split));
			node.right = new CGMTreeNode(node, ClassificationAndRegressionTree.getUpperPortion(node.data, n_split));
//			CGMTreeNode.DebugNode(node);
			return true; //we did actually do a split
		}
	}


	private void markNodeAsLeaf(CGMTreeNode node) {
		node.isLeaf = true;
//		System.out.println("marked " + node.stringID() + " as leaf");
		
//		System.out.print("LEAF: [");
//		int[] ns = countCompartments(node.data);
//		for (int k = 0; k < K; k++){
//			System.out.print(ns[k] + ",");
//		}
//		System.out.print("] of " + node.data.size() + "\n");
	}

	/**
	 * According to CGM98 p398, we uniformly sample one from the population
	 * 
	 * @param node 	the node to create a split attribute for
	 * @return 		the index of the split attribute to use
	 */
	public Integer assignSplitAttribute(CGMTreeNode node) {
		////////////////WRONG!!!!! see 3.2 of CGM98... although I can't imagine this matters too much
		//we're just building the skeleton of an initial tree that will change,
		//what we do here does not affect the MH / Gibbs sampling that comes later
		
		//create a set of possibilities
//		HashSet<Integer> already_used_predictors = node.getAlreadyUsedPredictors();
		ArrayList<Integer> could_be_used = new ArrayList<Integer>();
		for (int j = 0; j < p; j++){
//			//remove those that have already been used
//			if (!already_used_predictors.contains(j)){
				could_be_used.add(j);
//			}
		}		
//		if (could_be_used.size() == 0){
//			//if we got none, let the function know
//			return null;
//		}
//		else {
			//now choose one at random
			return could_be_used.get((int) Math.floor(Math.random() * could_be_used.size()));
//		}
	}
	
	/**
	 * According to CGM98 p398-9, after we have chosen a split attribute, we create a split
	 * rule $x < x^*$ based on that attribute by just uniformly sampling one data point from 
	 * the **residual** design matrix for that predictor, i.e. sample from $\x_{\cdot, j}$
	 * where $\x$ is what's left over at this junction from the original data
	 * 
	 * @param data	the left over data that is under consideration at this node, a subset of the original design matrix 
	 * 
	 * @param j		the attribute we are splitting on
	 * @return		the data point to split at
	 */
	public double assignSplitValue(List<double[]> data, int j) {
		int nsub = data.size();
		ArrayList<Double> predictor_data = new ArrayList<Double>(nsub);
		for (int i = 0; i < nsub; i++){
			predictor_data.add(data.get(i)[j]);
		}
		//now choose one at random
		return predictor_data.get((int) Math.floor(Math.random() * nsub));	//coll.get_random_element()	
	}

	/**
	 * Calculates the probability that this node should be split
	 * 
	 * @param node	the node under question
	 * @return		the probability according to CGM98 p938
	 */
	public double calculateProbabilityOfSplitting(CGMTreeNode node) {
		return getAlpha() * Math.pow(1 + node.generation, -getBeta());
	}
	
	public double probabilityOfTree(CGMTreeNode root){
		return probabilityOfNode(root);
	}
	
	private double probabilityOfNode(CGMTreeNode node){
		double prob_splitting_this_node = calculateProbabilityOfSplitting(node);
		if (node.left != null && node.right != null){
			return prob_splitting_this_node * probabilityOfNode(node.left) * probabilityOfNode(node.right);
		}
		if (node.left != null && node.right == null){
			return prob_splitting_this_node * probabilityOfNode(node.left);
		}
		if (node.left == null && node.right != null){
			return prob_splitting_this_node * probabilityOfNode(node.right);
		}	
		return prob_splitting_this_node;
	}
}
