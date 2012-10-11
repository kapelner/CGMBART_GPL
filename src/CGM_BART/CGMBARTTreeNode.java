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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * A dumb struct to store information about a 
 * node in the Bayesian decision tree.
 * 
 * Unfortunately, this has parted ways too much with a CART
 * structure, that I needed to create something new
 * 
 * @author Adam Kapelner
 */
public class CGMBARTTreeNode implements Cloneable, Serializable {
	private static final long serialVersionUID = -5584590448078741112L;

	/** a link back to the overall bart model */
	private CGMBART_02_hyperparams cgmbart;	
	/** the parent node */
	public CGMBARTTreeNode parent;
	/** the left daughter node */
	public CGMBARTTreeNode left;
	/** the right daughter node */
	public CGMBARTTreeNode right;
	/** the generation of this node from the top node (root note has generation = 0 by definition) */
	public int depth;
	/** the amount of data points at this node */
	public int n_eta;
	/** is this node a terminal leaf? */
	public boolean isLeaf;
	/** the attribute this node makes a decision on */
	public Integer splitAttributeM;
	/** the value this node makes a decision on */
	public Double splitValue;
	/** if this is a leaf node, then the result of the classification, otherwise null */
	public Double klass;
	/** if this is a leaf node, then the result of the prediction for regression, otherwise null */
	public Double y_prediction;
	/** the remaining data records at this point in the tree construction (freed after tree is constructed) */
	public transient List<double[]> data;	

	public CGMBARTTreeNode(CGMBARTTreeNode parent, List<double[]> data, CGMBART_02_hyperparams cgmbart){
		this.parent = parent;
		this.data = clone_data_matrix_with_new_y_optional(data, null);
		this.cgmbart = cgmbart;
		if (data != null){
			n_eta = data.size();
		}
		if (parent != null){
			depth = parent.depth + 1;
		}
		isLeaf = true; //default is that it is a leaf
	}
	
	public CGMBARTTreeNode(CGMBARTTreeNode parent){
		this(parent, null, parent.cgmbart);
	}
	
	public CGMBARTTreeNode clone(){
		return clone(false);
	}
	
	public double[] responses(){
		double[] ys = new double[n_eta];
		for (int i = 0; i < n_eta; i++){
			double[] record = data.get(i);
			ys[i] = record[record.length - 1];
		}
//		System.out.println("ys: " + Tools.StringJoin(ys, ", "));
		return ys;
	}
	
	public HashSet<Double> responsesAsHash(){
		double[] ys = responses();
		HashSet<Double> responses = new HashSet<Double>(ys.length);
		for (int i = 0; i < ys.length; i++){
			responses.add(ys[i]);
		}
		return responses;
	}
	
	public double avgResponse(){
		return StatToolbox.sample_average(responses());
	}
	
	/** clones this node (if you clone the root, you clone the entire tree) */
	public CGMBARTTreeNode clone(boolean clone_data){ //"data" element always null in clone
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
		CGMBARTTreeNode copy = new CGMBARTTreeNode(parent, new_data, cgmbart);
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
		copy.n_eta = n_eta;
//		copy.predictors_that_can_be_assigned = predictors_that_can_be_assigned;
//		copy.possible_split_values = possible_split_values;
		return copy;
	}
	
	//toolbox functions
	public static ArrayList<CGMBARTTreeNode> getTerminalNodesWithDataAboveOrEqualToN(CGMBARTTreeNode node, int n_rule){
		ArrayList<CGMBARTTreeNode> terminal_nodes_data_above_n = new ArrayList<CGMBARTTreeNode>();
		findTerminalNodesDataAboveOrEqualToN(node, terminal_nodes_data_above_n, n_rule);
		return terminal_nodes_data_above_n;
	}
	
	public ArrayList<CGMBARTTreeNode> getTerminalNodes(){
		return getTerminalNodesWithDataAboveOrEqualToN(this, 0);
	}
	
	private static void findTerminalNodesDataAboveOrEqualToN(CGMBARTTreeNode node, ArrayList<CGMBARTTreeNode> terminal_nodes, int n_rule) {
		if (node.isLeaf && node.data.size() >= n_rule){
			terminal_nodes.add(node);
		}
		else if (!node.isLeaf){ //as long as we're not in a leaf we should recurse
			if (node.left == null || node.right == null){
				System.err.println("error node no children during findTerminalNodesDataAboveOrEqualToN id: " + node.stringID());
//				DebugNode(node);
			}			
			findTerminalNodesDataAboveOrEqualToN(node.left, terminal_nodes, n_rule);
			findTerminalNodesDataAboveOrEqualToN(node.right, terminal_nodes, n_rule);
		}
	}

//	public static int numTerminalNodes(CGMBARTTreeNode node){
//		return getTerminalNodesWithDataAboveOrEqualToN(node, 0).size();
//	}
//	
//	public static int numTerminalNodesDataAboveN(CGMBARTTreeNode node, int n_rule){
//		return getTerminalNodesWithDataAboveOrEqualToN(node, n_rule).size();
//	}
	
	public ArrayList<CGMBARTTreeNode> getPrunableAndChangeableNodes(){
		ArrayList<CGMBARTTreeNode> prunable_and_changeable_nodes = new ArrayList<CGMBARTTreeNode>();
		findPrunableAndChangeableNodes(this, prunable_and_changeable_nodes);
		return prunable_and_changeable_nodes;
	}

	private static void findPrunableAndChangeableNodes(CGMBARTTreeNode node, ArrayList<CGMBARTTreeNode> prunable_nodes) {
		if (node.isLeaf){
			return;
		}
		else if (node.left.isLeaf && node.right.isLeaf){
			prunable_nodes.add(node);
		}
		else {
			findPrunableAndChangeableNodes(node.left, prunable_nodes);
			findPrunableAndChangeableNodes(node.right, prunable_nodes);
		}
	}

	/**
	 * We prune the tree at this node. We cut off its children, mark is as a leaf,
	 * and delete its split rule
	 * 
	 * @param node	the node at which to trim the tree at
	 */
	public static void pruneTreeAt(CGMBARTTreeNode node) {
		node.left = null;
		node.right = null;
		node.isLeaf = true;
		node.splitAttributeM = null;
		node.splitValue = null;
	}

//	public static HashSet<CGMBARTTreeNode> selectBranchNodesWithTwoLeaves(ArrayList<CGMBARTTreeNode> terminalNodes) { 
//		HashSet<CGMBARTTreeNode> branch_nodes = new HashSet<CGMBARTTreeNode>();
//		for (CGMBARTTreeNode node : terminalNodes){
//			if (node.parent == null){
//				continue;
//			}
//			if (node.parent.left.isLeaf && node.parent.right.isLeaf){
//				branch_nodes.add(node.parent);
//			}
//		}
//		return branch_nodes;
//	}
	
//	public static ArrayList<CGMBARTTreeNode> findInternalNodes(CGMBARTTreeNode root){
//		ArrayList<CGMBARTTreeNode> internal_nodes = new ArrayList<CGMBARTTreeNode>();
//		findInternalNodesRecursively(root, internal_nodes);
//		return internal_nodes;
//	}
//
//	private static void findInternalNodesRecursively(CGMBARTTreeNode node, ArrayList<CGMBARTTreeNode> internal_nodes) {
//		//if we are a leaf, get out
//		if (node.isLeaf){
//			return;
//		}
//		internal_nodes.add(node);
//		//recurse to find others
//		findInternalNodesRecursively(node.left, internal_nodes);
//		findInternalNodesRecursively(node.right, internal_nodes);
//	}

//	public static ArrayList<CGMBARTTreeNode> findDoublyInternalNodes(CGMBARTTreeNode root) {
//		ArrayList<CGMBARTTreeNode> internal_nodes = findInternalNodes(root);
//		ArrayList<CGMBARTTreeNode> doubly_internal_nodes = new ArrayList<CGMBARTTreeNode>();
//		//remove all those whose children aren't also internal nodes
//		for (CGMBARTTreeNode node : internal_nodes){
//			if (!node.isLeaf && node.parent != null && !node.right.isLeaf && !node.left.isLeaf){
//				doubly_internal_nodes.add(node);
//			}
//		}
//		return doubly_internal_nodes;
//	}

	public double classification_or_regression_prediction() {
		if (klass == null){
//			System.out.println("evaluate " + y_prediction);
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

//	public int widestGeneration() {
//		HashMap<Integer, Integer> generation_to_freq = new HashMap<Integer, Integer>();
//		this.widestGeneration(generation_to_freq);
//		Object[] thicknesses = generation_to_freq.values().toArray();
//		//now go through and get the maximum
//		int max_thickness = Integer.MIN_VALUE;
//		for (int i = 0; i < thicknesses.length; i++){
//			if ((Integer)thicknesses[i] > max_thickness){
//				max_thickness = (Integer)thicknesses[i];
//			}
//		}
//		return max_thickness;
//	}
//	public void widestGeneration(HashMap<Integer, Integer> generation_to_freq) {
//		//first do the incrementation
//		if (generation_to_freq.containsKey(this.generation)){
//			//now we check if this node has children to make it thicker
//			if (this.isLeaf){
//				generation_to_freq.put(this.generation, generation_to_freq.get(this.generation) + 1);
//			}
//			else {
//				generation_to_freq.put(this.generation, generation_to_freq.get(this.generation) + 2);
//			}
//		}
//		else {
//			generation_to_freq.put(this.generation, 1);
//		}
//		//if it's a leaf, we're done
//		if (this.isLeaf){
//			return;
//		}		
//		//otherwise, recurse through the children
//		this.left.widestGeneration(generation_to_freq);
//		this.right.widestGeneration(generation_to_freq);
//	}

	public double Evaluate(double[] xs) {
		CGMBARTTreeNode evalNode = this;
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
			if (xs[evalNode.splitAttributeM] <= evalNode.splitValue){
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
	
	public static void propagateDataByChangedRule(CGMBARTTreeNode node, boolean clean_cache) {
		//only propagate if we are in a split node and NOT a leaf
		if (node.isLeaf){
			return;
		}
//		System.out.println("propagate changes " + node.stringID() + "  new n:" + node.n);

//		System.out.println("propagate changes in node that is not leaf");
		//split the data correctly
		ClassificationAndRegressionTree.SortAtAttribute(node.data, node.splitAttributeM);
		int n_split = ClassificationAndRegressionTree.getSplitPoint(node.data, node.splitAttributeM, node.splitValue);
		//now set the data in the offspring
		node.left.data = ClassificationAndRegressionTree.getLowerPortion(node.data, n_split);			
		node.left.n_eta = node.left.data.size();
//			for (int i = 0; i < node.left.n; i++){
//				System.out.println("parent " + node.stringID() + " left node " + node.left.stringID()+ " record num " + (i+1) + " " + IOTools.StringJoin(node.left.data.get(i), ","));
//			}
//			System.out.println("left avg: " + StatToolbox.sample_average(node.left.get_ys_in_data())); 
		node.right.data = ClassificationAndRegressionTree.getUpperPortion(node.data, n_split);
		node.right.n_eta = node.right.data.size();
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
			propagateDataByChangedRule(node.left, clean_cache);
			propagateDataByChangedRule(node.right, clean_cache);
//			}
	}

	//////CHECK THIS LATER
	public void updateWithNewResponsesAndPropagate(ArrayList<double[]> X_y, double[] y_new, int p) {
		//set the root node data
		this.data = clone_data_matrix_with_new_y_optional(X_y, y_new);
		//now just propagate away
		propagateDataByChangedRule(this, true);
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
	
//	public Double get_pred_for_nth_leaf(int leaf_num) {
//		String leaf_num_binary = Integer.toBinaryString(leaf_num);
//		//now hack off first digit
//		leaf_num_binary = leaf_num_binary.substring(1, leaf_num_binary.length());
//
////		System.out.println("get_pred_for_nth_leaf gen: directions: " + new String(leaf_num_binary));
//		
//		//now that we have our direction array, now we just iterate down the line, begin right where we are
//		CGMBARTTreeNode node = this;
//		for (char direction : leaf_num_binary.toCharArray()){
//			if (direction == '0'){
//				node = node.left;
//			}
//			else {
//				node = node.right;
//			}
//			//if this node does not exist in our tree, we're done
//			if (node == null){
//				return null;
//			}
//		}
//		return node.y_prediction;
//	}
	
	//CHECK as well
	public Double prediction_untransformed(){
		if (y_prediction == null){
			return null;
		}
		return cgmbart.un_transform_y(y_prediction);
	}
	
	public double avg_response_untransformed(){
		return cgmbart.un_transform_y(avgResponse());
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

	public LinkedHashMap<CGMBARTTreeNode, String> getLineage() {
		LinkedHashMap<CGMBARTTreeNode, String> lineage = new LinkedHashMap<CGMBARTTreeNode, String>();
		CGMBARTTreeNode node = this;
		while (true){
			CGMBARTTreeNode oldnode = node;
			node = node.parent;
			if (node == null){
				break;
			}			
			lineage.put(node, oldnode == node.left ? "L" : "R");
		}
		return lineage;
	}

	public double sumResponsesQuantitySqd() {
		return Math.pow(sumResponses(), 2);
	}

	public double sumResponses() {
		double sum = 0;
		for (int i = 0; i < n_eta; i++){
			double[] record = data.get(i);
			sum += record[record.length - 1];
		}
		return sum;
	}	
		
	
	/**
	 * Gets the total number of predictors that could be used for rules at this juncture
	 * @return
	 */
	public int pAdj(){
		return predictorsThatCouldBeUsedToSplitAtNode().size();
	}
	
	protected ArrayList<Integer> predictorsThatCouldBeUsedToSplitAtNode() {
		ArrayList<Integer> possible_rule_variables = new ArrayList<Integer>();
		
		for (int j = 0; j < cgmbart.getP(); j++){
			//if size of unique of x_i > 1
			ArrayList<Double> x_dot_j = Classifier.getColVector(data, j);
			HashSet<Double> unique_x_dot_j = new HashSet<Double>(x_dot_j);
			if (unique_x_dot_j.size() >= 2){
				possible_rule_variables.add(j);
			}
		}	
		
		return possible_rule_variables;
	}

	/**
	 * Gets the total number of split points that can be used for rules at this juncture
	 * @return
	 */
	public int nAdj(){
		return possibleSplitValuesGivenAttribute().size();
	}	
	
	protected ArrayList<Double> possibleSplitValuesGivenAttribute() {
		ArrayList<Double> x_dot_j = Classifier.getColVector(data, splitAttributeM);
		Double max = Collections.max(x_dot_j);
		HashSet<Double> unique_x_dot_j = new HashSet<Double>(x_dot_j);
		unique_x_dot_j.remove(max);
		
		return new ArrayList<Double>(unique_x_dot_j);
	}

	/**
	 * Pick a random predictor from the set of possible predictors at this juncture
	 * @return
	 */
	public int pickRandomPredictorThatCanBeAssigned(){
		ArrayList<Integer> predictors = predictorsThatCouldBeUsedToSplitAtNode();
		return predictors.get((int)Math.floor(StatToolbox.rand() * predictors.size()));
	}
	
	public Double pickRandomSplitValue() {
		ArrayList<Double> split_values = possibleSplitValuesGivenAttribute();
		return split_values.get((int) Math.floor(StatToolbox.rand() * split_values.size()));
	}

	public boolean isStump() {
		return parent == null && left == null && right == null;
	}

	
//	private static void DebugWholeTreeRecursively(CGMBARTTreeNode node){
//		DebugNode(node);
//		if (node.left != null){
//			DebugWholeTreeRecursively(node.left);
//		}
//		if (node.right != null){
//			DebugWholeTreeRecursively(node.right);
//		}		
//	}	
//	
//	public static void DebugWholeTree(CGMBARTTreeNode root){
//		System.out.println("DEBUG WHOLE TREE ***********************************");
//		DebugWholeTreeRecursively(root);
//		System.out.println("****************************************************");
//	}	
//	
//	public static void DebugNode(CGMBARTTreeNode node){	
//		System.out.println(" DEBUG: " + node.stringLocation(true) + "   ID: " + node.stringID());
//		if (node.data != null){
//			System.out.println("  data n: " + node.data.size());
//		}	
//		if (node.parent != null){
//			System.out.println("  parent: " + node.parent.stringID());
//		}
//		else {
//			System.out.println("  parent: null");
//		}
//		if (node.left != null){
//			System.out.println("  left: " + node.left.stringID() + "\t\tnL: " + node.left.data.size());
//		}
//		if (node.right != null){
//			System.out.println("  right: " + node.right.stringID() + "\t\tnR: " + node.right.data.size());
//		}
//		System.out.println("  gen: " + node.generation);
//		if (node.isLeaf){
//			System.out.println("  isleaf");
//		}
//
//		if (node.splitAttributeM != null){
//			System.out.println("  splitAtrr: " + node.splitAttributeM);
//		}
//		if (node.splitValue != null){
//			System.out.println("  splitval: " + node.splitValue);
//		}
//		if (node.klass != null){
//			System.out.println("  class: " + node.klass);
//		}
//	}
	
	public String stringID() {
		return toString().split("@")[1];
	}	
	
//	public double[] get_ys_in_data(){
//		double[] ys = new double[data.size()];
//		for (int i = 0; i < data.size(); i++){
//			double[] record = data.get(i);
//			ys[i] = record[record.length - 1];
//		}
//		return ys;
//	}
	
	//serializable happy
	public CGMBARTTreeNode getLeft() {
		return left;
	}
	public CGMBARTTreeNode getRight() {
		return right;
	}
	public int getGeneration() {
		return depth;
	}
	public void setGeneration(int generation) {
		this.depth = generation;
	}	
	public boolean isLeaf() {
		return isLeaf;
	}
	public void setLeaf(boolean isLeaf) {
		this.isLeaf = isLeaf;
	}
	public void setLeft(CGMBARTTreeNode left) {
		this.left = left;
	}
	public void setRight(CGMBARTTreeNode right) {
		this.right = right;
	}
	public int getSplitAttributeM() {
		return splitAttributeM;
	}
	public void setSplitAttributeM(int splitAttributeM) {
		this.splitAttributeM = splitAttributeM;
	}
	public double getSplitValue() {
		return splitValue;
	}
	public void setSplitValue(double splitValue) {
		this.splitValue = splitValue;
	}
	public Double getKlass() {
		return klass;
	}
	public void setKlass(Double klass) {
		this.klass = klass;
	}
	public int getNEta() {
		return n_eta;
	}
	public void setNEta(int n_eta) {
		this.n_eta = n_eta;
	}

	public int numTimesAttrUsed(int j) {
		if (this.isLeaf){
			return 0;
		}
		return (this.splitAttributeM == j ? 1 : 0) + this.left.numTimesAttrUsed(j) + this.right.numTimesAttrUsed(j);
	}

//	public CGMBARTTreeNode findCorrespondingNodeOnSimilarTree(CGMBARTTreeNode node) {
//		char[] location = node.stringLocation(true).toCharArray();
//		CGMBARTTreeNode corresponding_node = this;
//		if (location[0] == 'P' && location.length == 1){
//			return corresponding_node;
//		}
//		
//		for (int i = 0; i < location.length; i++){
//			if (location[i] == 'L'){
//				corresponding_node = corresponding_node.left;
//			}
//			else if (location[i] == 'R'){
//				corresponding_node = corresponding_node.right;
//			}			
//		}
//		
//		return corresponding_node;
//	}
}