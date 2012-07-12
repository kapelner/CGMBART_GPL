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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import CGM_BART.CGMBART;

/**
 * A dumb struct to store information about a 
 * node in the Bayesian decision tree.
 * 
 * Unfortunately, this has parted ways too much with a CART
 * structure, that I needed to create something new
 * 
 * @author Adam Kapelner
 */
public class CGMTreeNode extends TreeNode implements Cloneable, Serializable {
	private static final long serialVersionUID = -5584590448078741112L;

	/** a link back to the overall bart model */
	private CGMBART bart;
	/** the parent node */
	public CGMTreeNode parent;
	/** the left daughter node */
	public CGMTreeNode left;
	/** the right daughter node */
	public CGMTreeNode right;
	/** the generation of this node from the top node (root note has generation = 0 by definition) */
	public int generation;
	/** the amount of data points at this node */
	public int n;
	/** keep the log proportional likelihood of this tree around, if null, must be computed */
	public Double log_prop_lik;

	public ArrayList<Integer> predictors_that_can_be_assigned;
	public ArrayList<Double> possible_split_values;


	public CGMTreeNode(CGMTreeNode parent, List<double[]> data, CGMBART bart){
		this.parent = parent;
		this.data = clone_data_matrix_with_new_y_optional(data, null);
		this.bart = bart;
		if (data != null){
			n = data.size();
		}
		if (parent != null){
			generation = parent.generation + 1;
		}
		isLeaf = true; //default is that it is a leaf
	}
	
	public CGMTreeNode(CGMTreeNode parent, List<double[]> data){
		this(parent, data, parent.bart);
	}	
	
	public CGMTreeNode clone(){
		return clone(false);
	}
	
	public double[] responses(){
		double[] ys = new double[n];
		for (int i = 0; i < n; i++){
			double[] record = data.get(i);
			ys[i] = record[record.length - 1];
		}
		return ys;
	}
	
	public double avgResponse(){
		return StatToolbox.sample_average(responses());
	}
	
	/** clones this node (if you clone the root, you clone the entire tree) */
	public CGMTreeNode clone(boolean clone_data){ //"data" element always null in clone
		List<double[]> new_data = new ArrayList<double[]>(data == null ? 0 : data.size());
		//first clone the data
		if (clone_data && data != null){
			for (double[] record : data){
				double[] new_record = new double[record.length];
				for (int i = 0; i < record.length; i++){
					new_record[i] = record[i];
				}
				new_data.add(new_record);
			}
		}
		else {
			new_data = data;
		}
		CGMTreeNode copy = new CGMTreeNode(parent, new_data, bart);
		copy.isLeaf = isLeaf;
		if (left != null){ //we need to clone the child and mark parent correctly
			copy.left = left.clone(clone_data);
			copy.left.parent = copy;
		}
		if (right != null){ //we need to clone the child and mark parent correctly
			copy.right = right.clone(clone_data);
			copy.right.parent = copy;
		}
		copy.splitAttributeM = splitAttributeM;
		copy.splitValue = splitValue;
		copy.klass = klass;	
		copy.n = n;
		copy.log_prop_lik = log_prop_lik;
		copy.predictors_that_can_be_assigned = predictors_that_can_be_assigned;
		copy.possible_split_values = possible_split_values;
		return copy;
	}

	public HashSet<Integer> getAlreadyUsedPredictors() {
		HashSet<Integer> already_used_predictors = new HashSet<Integer>();
		//start with this current node
		CGMTreeNode node = this;
		do {
			//snag the split attribute
			already_used_predictors.add(node.splitAttributeM);
			//jump to parent and play it again 
			node = node.parent;
		} while (node != null); //if we reached the root, bail
		
		return already_used_predictors;
	}
	
	public static void DebugWholeTree(CGMTreeNode root){
		System.out.println("DEBUG WHOLE TREE ***********************************");
		DebugWholeTreeRecursively(root);
		System.out.println("****************************************************");
	}
	
	private static void DebugWholeTreeRecursively(CGMTreeNode node){
		DebugNode(node);
		if (node.left != null){
			DebugWholeTreeRecursively(node.left);
		}
		if (node.right != null){
			DebugWholeTreeRecursively(node.right);
		}		
	}	
	
	public static void DebugNode(CGMTreeNode node){	
		System.out.println(" DEBUG: " + node.stringID());
		if (node.data != null){
			System.out.println("  data n: " + node.data.size());
		}	
		if (node.parent != null){
			System.out.println("  parent: " + node.parent.stringID());
		}
		else {
			System.out.println("  parent: null");
		}
		if (node.left != null){
			System.out.println("  left: " + node.left.stringID() + "\t\tnL: " + node.left.data.size());
		}
		if (node.right != null){
			System.out.println("  right: " + node.right.stringID() + "\t\tnR: " + node.right.data.size());
		}
		System.out.println("  gen: " + node.generation);
		if (node.isLeaf){
			System.out.println("  isleaf");
		}

		if (node.splitAttributeM != null){
			System.out.println("  splitAtrr: " + node.splitAttributeM);
		}
		if (node.splitValue != null){
			System.out.println("  splitval: " + node.splitValue);
		}
		if (node.klass != null){
			System.out.println("  class: " + node.klass);
		}
	}
	
	//serializable happy
	public CGMTreeNode getLeft() {
		return left;
	}
	public CGMTreeNode getRight() {
		return right;
	}
	public int getGeneration() {
		return generation;
	}
	public void setGeneration(int generation) {
		this.generation = generation;
	}

	
	//toolbox functions
	public static ArrayList<CGMTreeNode> getTerminalNodesWithDataAboveN(CGMTreeNode node, int n_rule){
		ArrayList<CGMTreeNode> terminal_nodes_data_above_n = new ArrayList<CGMTreeNode>();
		findTerminalNodesDataAboveN(node, terminal_nodes_data_above_n, n_rule);
		return terminal_nodes_data_above_n;
	}
	
	public ArrayList<CGMTreeNode> getTerminalNodes(){
		return getTerminalNodesWithDataAboveN(this, 0);
	}
	
	private static void findTerminalNodesDataAboveN(CGMTreeNode node, ArrayList<CGMTreeNode> terminal_nodes, int n_rule) {
		if (node.isLeaf && node.data.size() >= n_rule){
			terminal_nodes.add(node);
		}
		else if (!node.isLeaf){ //as long as we're not in a leaf we should recurse
			if (node.left == null || node.right == null){
				System.err.println("error node: " + node.stringID());
				DebugNode(node);
			}			
			findTerminalNodesDataAboveN(node.left, terminal_nodes, n_rule);
			findTerminalNodesDataAboveN(node.right, terminal_nodes, n_rule);
		}
	}

	public static int numTerminalNodes(CGMTreeNode node){
		return getTerminalNodesWithDataAboveN(node, 0).size();
	}
	
	public static int numTerminalNodesDataAboveN(CGMTreeNode node, int n_rule){
		return getTerminalNodesWithDataAboveN(node, n_rule).size();
	}
	
	public static ArrayList<CGMTreeNode> getPrunableNodes(CGMTreeNode node){
		ArrayList<CGMTreeNode> prunable_nodes = new ArrayList<CGMTreeNode>();
		findPrunableNodes(node, prunable_nodes);
		return prunable_nodes;
	}	

	private static void findPrunableNodes(CGMTreeNode node, ArrayList<CGMTreeNode> prunable_nodes) {
		if (node.isLeaf){
			return;
		}
		else if (node.left.isLeaf && node.right.isLeaf){
			prunable_nodes.add(node);
		}
		else {
			findPrunableNodes(node.left, prunable_nodes);
			findPrunableNodes(node.right, prunable_nodes);
		}
	}

	/**
	 * We prune the tree at this node. We cut off its children, mark is as a leaf,
	 * and delete its split rule
	 * 
	 * @param node	the node at which to trim the tree at
	 */
	public static void pruneTreeAt(CGMTreeNode node) {
		node.left = null;
		node.right = null;
		node.isLeaf = true;
		node.splitAttributeM = null;
		node.splitValue = null;
	}

	public static HashSet<CGMTreeNode> selectBranchNodesWithTwoLeaves(ArrayList<CGMTreeNode> terminalNodes) { HashSet<CGMTreeNode> branch_nodes = new HashSet<CGMTreeNode>();
		for (CGMTreeNode node : terminalNodes){
			if (node.parent == null){
				continue;
			}
			if (node.parent.left.isLeaf && node.parent.right.isLeaf){
				branch_nodes.add(node.parent);
			}
		}
		return branch_nodes;
	}
	
	public static ArrayList<CGMTreeNode> findInternalNodes(CGMTreeNode root){
		ArrayList<CGMTreeNode> internal_nodes = new ArrayList<CGMTreeNode>();
		findInternalNodesRecursively(root, internal_nodes);
		return internal_nodes;
	}

	private static void findInternalNodesRecursively(CGMTreeNode node, ArrayList<CGMTreeNode> internal_nodes) {
		//if we are a leaf, get out
		if (node.isLeaf){
			return;
		}
		internal_nodes.add(node);
		//recurse to find others
		findInternalNodesRecursively(node.left, internal_nodes);
		findInternalNodesRecursively(node.right, internal_nodes);
	}

	public static ArrayList<CGMTreeNode> findDoublyInternalNodes(CGMTreeNode root) {
		ArrayList<CGMTreeNode> internal_nodes = findInternalNodes(root);
		ArrayList<CGMTreeNode> doubly_internal_nodes = new ArrayList<CGMTreeNode>();
		//remove all those whose children aren't also internal nodes
		for (CGMTreeNode node : internal_nodes){
			if (!node.isLeaf && node.parent != null && !node.right.isLeaf && !node.left.isLeaf){
				doubly_internal_nodes.add(node);
			}
		}
		return doubly_internal_nodes;
	}

	public double classification_or_regression_prediction() {
		if (klass == null){
			return y_prediction;
		}
		return klass;
	}
	
	public int deepestNode(){
		if (this.isLeaf){
			return 0;
		}
		else {
			int ldn = this.left.deepestNode();
			int rdn = this.right.deepestNode();
			if (ldn > rdn){
				return 1 + ldn;
			}
			else {
				return 1 + rdn;
			}
		}
	}

	public int widestGeneration() {
		HashMap<Integer, Integer> generation_to_freq = new HashMap<Integer, Integer>();
		this.widestGeneration(generation_to_freq);
		Object[] thicknesses = generation_to_freq.values().toArray();
		//now go through and get the maximum
		int max_thickness = Integer.MIN_VALUE;
		for (int i = 0; i < thicknesses.length; i++){
			if ((Integer)thicknesses[i] > max_thickness){
				max_thickness = (Integer)thicknesses[i];
			}
		}
		return max_thickness;
	}
	public void widestGeneration(HashMap<Integer, Integer> generation_to_freq) {
		//first do the incrementation
		if (generation_to_freq.containsKey(this.generation)){
			//now we check if this node has children to make it thicker
			if (this.isLeaf){
				generation_to_freq.put(this.generation, generation_to_freq.get(this.generation) + 1);
			}
			else {
				generation_to_freq.put(this.generation, generation_to_freq.get(this.generation) + 2);
			}
		}
		else {
			generation_to_freq.put(this.generation, 1);
		}
		//if it's a leaf, we're done
		if (this.isLeaf){
			return;
		}		
		//otherwise, recurse through the children
		this.left.widestGeneration(generation_to_freq);
		this.right.widestGeneration(generation_to_freq);
	}

	//serializable happy
	public int getN() {
		return n;
	}

	public void setN(int n) {
		this.n = n;
	}

	public double Evaluate(double[] record) {
		CGMTreeNode evalNode = this;
//		System.out.println("Evaluate record: " + IOTools.StringJoin(record, " "));
		while (true){
//			System.out.println("evaluate via node: " + evalNode.stringID());
//			CGMTreeNode.DebugNode((CGMTreeNode)evalNode);
			if (evalNode.isLeaf){
				return evalNode.classification_or_regression_prediction();
			}
			//all split rules are less than or equals (this is merely a convention)
			//it's a convention that makes sense - if X_.j is binary, and the split values can only be 0/1
			//then it MUST be <= so both values can be considered
//			System.out.println("Rule: X_" + evalNode.splitAttributeM + " < " + evalNode.splitValue + " ie " + record[evalNode.splitAttributeM] + " < " + evalNode.splitValue);
			if (record[evalNode.splitAttributeM] <= evalNode.splitValue){
//				System.out.println("went left");
				evalNode = evalNode.left;
			}
			else {
//				System.out.println("went right");
				evalNode = evalNode.right;
			}
		}
	}

	public void flushNodeData() {
//		System.out.println("FlushNodeData");
//		CGMTreeNode.DebugNode(node); 
		this.data = null;
		if (this.left != null)
			this.left.flushNodeData();
		if (this.right != null)
			this.right.flushNodeData();
	}
	
	public static void propagateRuleChangeOrSwapThroughoutTree(CGMTreeNode node, boolean clean_cache) {
		//only propagate if we are in a split node and NOT a leaf
		if (node.left == null || node.right == null){
			return;
		}
//		System.out.println("propagate changes " + node.stringID() + "  new n:" + node.n);
		if (!node.isLeaf){
//			System.out.println("propagate changes in node that is not leaf");
			//split the data correctly
			ClassificationAndRegressionTree.SortAtAttribute(node.data, node.splitAttributeM);
			int n_split = ClassificationAndRegressionTree.getSplitPoint(node.data, node.splitAttributeM, node.splitValue);
			//now set the data in the offspring
			node.left.data = ClassificationAndRegressionTree.getLowerPortion(node.data, n_split);			
			node.left.n = node.left.data.size();
//			for (int i = 0; i < node.left.n; i++){
//				System.out.println("parent " + node.stringID() + " left node " + node.left.stringID()+ " record num " + (i+1) + " " + IOTools.StringJoin(node.left.data.get(i), ","));
//			}
//			System.out.println("left avg: " + StatToolbox.sample_average(node.left.get_ys_in_data())); 
			node.right.data = ClassificationAndRegressionTree.getUpperPortion(node.data, n_split);
			node.right.n = node.right.data.size();
//			for (int i = 0; i < node.right.n; i++){
//				System.out.println("parent " + node.stringID() + " right node " + node.right.stringID()+ " record num " + (i+1) + " " + IOTools.StringJoin(node.right.data.get(i), ","));
//			}		
//			System.out.println("right avg: " + StatToolbox.sample_average(node.right.get_ys_in_data())); 
			///////////////////////////NO PRUNING!!!!
//			if (node.left.n == 0 || node.right.n == 0){
//				//gotta prune if one of the children is empty
//				CGMTreeNode.pruneTreeAt(node);
//			}
//			else {
				//now recursively take care of the children
				propagateRuleChangeOrSwapThroughoutTree(node.left, clean_cache);
				propagateRuleChangeOrSwapThroughoutTree(node.right, clean_cache);
//			}
		}
	}

	public void updateWithNewResponsesAndPropagate(ArrayList<double[]> X_y, double[] y_new, int p) {
		//set the root node data
		this.data = clone_data_matrix_with_new_y_optional(X_y, y_new);
		//now just propagate away
		propagateRuleChangeOrSwapThroughoutTree(this, true);
	}
	
	public static ArrayList<double[]> clone_data_matrix_with_new_y_optional(List<double[]> X_y, double[] y_new){
		if (X_y == null){
			return null;
		}
		ArrayList<double[]> X_y_new = new ArrayList<double[]>(X_y.size());
		for (int i = 0; i < X_y.size(); i++){
			double[] original_record = X_y.get(i);
			int p = original_record.length - 1;
			double[] new_record = new double[p + 1];
			for (int j = 0; j <= p; j++){
				if (j == p && y_new != null){
					new_record[j] = y_new[i];
				}
				else {
					new_record[j] = original_record[j];
				}
			}
			X_y_new.add(new_record);

//			System.out.println("original_record: " + IOTools.StringJoin(original_record, ","));
//			System.out.println("new_record: " + IOTools.StringJoin(new_record, ","));
		}
		return X_y_new; 
	}
	
	public int numLeaves(){
		if (this.isLeaf){
			return 1;
		}
		else {
			return this.left.numLeaves() + this.right.numLeaves();
		}
	}

	public int numPruneNodesAvailable() {
		if (this.isLeaf){
			return 0;
		}
		if (this.left.isLeaf && this.right.isLeaf){
			return 1;
		}
		return this.left.numPruneNodesAvailable() + this.right.numPruneNodesAvailable();
	}	

	public String splitToString() {
		if (this.isLeaf){
			String klass = this.klass == null ? "null" : TreeIllustration.two_digit_format.format(this.klass);
			return "leaf rule: " + klass;
		}
		else {
			String split = this.splitAttributeM == null ? "null" : TreeIllustration.two_digit_format.format(this.splitAttributeM);
			String value = this.splitValue == null ? "null" : TreeIllustration.two_digit_format.format(this.splitValue);
			return "x_" + split + "  <  " + value;
		}
	}

	public void initLogPropLik() {
		this.log_prop_lik = 0.0;
	}
	
	public Double get_pred_for_nth_leaf(int leaf_num) {
		String leaf_num_binary = Integer.toBinaryString(leaf_num);
		//now hack off first digit
		leaf_num_binary = leaf_num_binary.substring(1, leaf_num_binary.length());

//		System.out.println("get_pred_for_nth_leaf gen: directions: " + new String(leaf_num_binary));
		
		//now that we have our direction array, now we just iterate down the line, begin right where we are
		CGMTreeNode node = this;
		for (char direction : leaf_num_binary.toCharArray()){
			if (direction == '0'){
				node = node.left;
			}
			else {
				node = node.right;
			}
			//if this node does not exist in our tree, we're done
			if (node == null){
				return null;
			}
		}
		return node.y_prediction;
	}
	
	public Double prediction_untransformed(){
		if (y_prediction == null){
			return null;
		}
		return bart.un_transform_y(y_prediction);
	}
	
	public double avg_response_untransformed(){
		return bart.un_transform_y(avgResponse());
	}

	public String stringLocation(boolean show_parent) {
		if (this.parent == null){
			return show_parent ? "P" : "";
		}
		else if (this.parent.left == this){
			return this.parent.stringLocation(false) + "L";
		}
		else {
			return this.parent.stringLocation(false) + "R";
		}
	}

	public ArrayList<CGMTreeNode> getLineage() {
		ArrayList<CGMTreeNode> lineage = new ArrayList<CGMTreeNode>();
		CGMTreeNode node = this;
		while (true){
			node = node.parent;
			if (node == null){
				break;
			}			
			lineage.add(node);
		}
		return lineage;
	}

	public double sumResponsesSqd() {
		return Math.pow(sumResponses(), 2);
	}

	private double sumResponses() {
		double sum = 0;
		for (int i = 0; i < n; i++){
			double[] record = data.get(i);
			sum += record[record.length - 1];
		}
		return sum;
	}
	
	public int pAdj(){
//		System.out.println("pAdj on node " + this.stringID());		
		return predictors_that_can_be_assigned.size();
	}
	
	public int pickRandomPredictorThatCanBeAssigned(){
		return predictors_that_can_be_assigned.get((int)Math.floor(StatToolbox.rand() * predictors_that_can_be_assigned.size()));
	}
	
	public int nAdj(){
		return possible_split_values.size();
	}	

	public Double pickRandomSplitValue() {
		return possible_split_values.get((int) Math.floor(StatToolbox.rand() * possible_split_values.size()));
	}

	public boolean isStump() {
		return parent == null && left == null && right == null;
	}

	
}