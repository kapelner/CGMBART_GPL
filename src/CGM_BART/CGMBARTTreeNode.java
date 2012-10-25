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

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.TDoubleHashSet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
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
	
	public static final boolean DEBUG_NODES = false;

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
	/** is this node a terminal leaf? */
	public boolean isLeaf;
	/** the attribute this node makes a decision on */
	public Integer splitAttributeM;
	/** the value this node makes a decision on */
	public double splitValue;
	/** if this is a leaf node, then the result of the classification, otherwise null */
	public Double klass;
	/** if this is a leaf node, then the result of the prediction for regression, otherwise null */
	protected static final double BAD_FLAG = -Double.MAX_VALUE;
	public double y_pred = BAD_FLAG;
	/** the remaining data records at this point in the tree construction cols: x_1, ..., x_p, y, index */
	public transient List<double[]> data;
	/** the number of data points */
	public transient int n_eta;
	/** these are the yhats in the correct order */
	public transient double[] yhats;

	//variables that get cached
	/** the indices in {0,1,...,n-1} of the data records in this node */
	protected transient int[] indicies;	
	/** the y's in this node */
	protected transient double[] responses;
	/** self-explanatory */
	private transient double sum_responses_qty_sqd;
	/** self-explanatory */
	private transient double sum_responses_qty;	
	/** this caches the possible split variables */
	private transient ArrayList<Integer> possible_rule_variables;
	/** this caches the possible split values BY variable */
	private transient HashMap<Integer, TDoubleHashSet> possible_split_vals_by_attr;
	/** this caches the number of possible split variables */
	private transient Integer padj;	
	
	public CGMBARTTreeNode(){}	

	public CGMBARTTreeNode(CGMBARTTreeNode parent, CGMBART_02_hyperparams cgmbart){
		this.parent = parent;
		this.yhats = parent.yhats;
		this.cgmbart = cgmbart;
		
		if (parent != null){
			depth = parent.depth + 1;
		}
		isLeaf = true; //default is that it is a leaf
	}
	
	public CGMBARTTreeNode(CGMBARTTreeNode parent){
		this(parent, parent.cgmbart);
	}
	
	public CGMBARTTreeNode(CGMBART_02_hyperparams cgmbart) {
		this.cgmbart = cgmbart;
		isLeaf = true;
		depth = 0;
	}

	public CGMBARTTreeNode clone(){
		CGMBARTTreeNode copy = new CGMBARTTreeNode();
		copy.cgmbart = cgmbart;
		copy.parent = parent;
		copy.isLeaf = isLeaf;
		copy.splitAttributeM = splitAttributeM;
		copy.splitValue = splitValue;
		copy.klass = klass;
		copy.possible_rule_variables = possible_rule_variables;
		copy.possible_split_vals_by_attr = possible_split_vals_by_attr;
		copy.depth = depth;
		//////do not copy y_pred
		//now do data stuff
		copy.data = data;
		copy.responses = responses;
		copy.indicies = indicies;
		copy.n_eta = n_eta;
		copy.yhats = yhats;
		
		if (left != null){ //we need to clone the child and mark parent correctly
			copy.left = left.clone();
			copy.left.parent = copy;
		}
		if (right != null){ //we need to clone the child and mark parent correctly
			copy.right = right.clone();
			copy.right.parent = copy;
		}		
		return copy;
	}
	
//	public int[] getIndices(){
//		if (indicies == null){
//			indicies = new int[data.size()];
//			for (int i = 0; i < data.size(); i++){
//				indicies[i] = (int) data.get(i)[cgmbart.p + 1];
//			}
//		}
//		return indicies;		
//	}
//	
//	public double[] getResponses(){
//		if (responses == null){			
//			responses = new double[data.size()];
//			for (int i = 0; i < data.size(); i++){
//				responses[i] = data.get(i)[cgmbart.p];
//			}
////			System.out.println("getResponses internal on " + this.stringLocation(true) + " " + Tools.StringJoin(responses));
//		}
//		return responses;
//	}
	
	public double avgResponse(){
		return StatToolbox.sample_average(responses);
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
		if (node.isLeaf && node.n_eta >= n_rule){
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
		node.splitValue = BAD_FLAG;
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
			return y_pred;
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
		data = null;
		yhats = null;
		indicies = null;	
		responses = null;
		possible_rule_variables = null;
		possible_split_vals_by_attr = null;
		
		if (this.left != null)
			this.left.flushNodeData();
		if (this.right != null)
			this.right.flushNodeData();
	}
	
	/**
	 * Sorts a data matrix by an attribute from lowest record to highest record
	 * 
	 * @param data			the data matrix to be sorted
	 * @param j				the attribute to sort on
	 */
//	@SuppressWarnings("unchecked")
//	protected void SortAtAttribute(){
//		Collections.sort(data, new AttributeComparator(splitAttributeM));
//		//update indicies after sort
//		for (int i = 0; i < n_eta; i++){
//			indicies[i] = (int)data.get(i)[cgmbart.p + 1];
//		}
//	}	
	
	public static void propagateDataByChangedRule(CGMBARTTreeNode node) {		
		if (node.isLeaf){ //only propagate if we are in a split node and NOT a leaf
			node.printNodeDebugInfo("propagateDataByChangedRule LEAF");
			return;
		}
		//split the data correctly
//		node.SortAtAttribute();
//		int n_split = ClassificationAndRegressionTree.getSplitPoint(node.data, node.splitAttributeM, node.splitValue);
		//now set the data in the offspring
		int p = node.cgmbart.p;
		ArrayList<double[]> data_left = new ArrayList<double[]>(node.n_eta);
		ArrayList<double[]> data_right = new ArrayList<double[]>(node.n_eta);
		TIntArrayList left_indices = new TIntArrayList(node.n_eta); 
		TIntArrayList right_indices = new TIntArrayList(node.n_eta);
		TDoubleArrayList left_responses = new TDoubleArrayList(node.n_eta);
		TDoubleArrayList right_responses = new TDoubleArrayList(node.n_eta);
		
		for (int i = 0; i < node.n_eta; i++){
			double[] datum = node.data.get(i);
			if (datum[node.splitAttributeM] <= node.splitValue){
				data_left.add(datum);
				left_indices.add(node.indicies[i]);
				left_responses.add(datum[p]);
			}
			else {
				data_right.add(datum);
				right_indices.add(node.indicies[i]);	
				right_responses.add(datum[p]);
			}
		}
		
		node.left.data = data_left;			
		node.left.n_eta = node.left.data.size();
		node.left.responses = left_responses.toArray();
		node.left.indicies = left_indices.toArray();	
		node.right.data = data_right;
		node.right.n_eta = node.right.data.size();
		node.right.responses = right_responses.toArray();
		node.right.indicies = right_indices.toArray();
		propagateDataByChangedRule(node.left);
		propagateDataByChangedRule(node.right);
	}
	
//	public void getYhatsByDataIndex(double[] y_hats_by_index){
//		if (this.isLeaf){
//			double y_hat = classification_or_regression_prediction();
////			System.out.println("getYhatsByDataIndex for " + this.stringLocation(true) + " yhat = " + y_hat + " n_eta = " + (this.data == null ? "NULL" : this.data.size()));
//			for (double[] datum : this.data){
//				y_hats_by_index[(int) datum[cgmbart.p + 1]] = y_hat;
//			}
//		}
//		else {
//			this.left.getYhatsByDataIndex(y_hats_by_index);
//			this.right.getYhatsByDataIndex(y_hats_by_index);
//		}
//	}

	//////CHECK THIS LATER
//	public void updateWithNewResponsesAndPropagate(ArrayList<double[]> X_y, double[] y_new, int p) {
//		//set the root node data
//		this.data = Classifier.clone_data_matrix_with_new_y_optional(X_y, y_new);
//		this.n_eta = this.data.size(); //make sure the parent has the right size
//		//now just propagate away
//		propagateDataByChangedRule(this, true);
//	}
	
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
			String value = this.splitValue == BAD_FLAG ? "null" : TreeIllustration.two_digit_format.format(this.splitValue);
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
	public double prediction_untransformed(){
		if (y_pred == BAD_FLAG){
			return BAD_FLAG;
		}
		return cgmbart.un_transform_y(y_pred);
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
		if (sum_responses_qty_sqd == 0){
			sum_responses_qty_sqd = Math.pow(sumResponses(), 2);
		}
		return sum_responses_qty_sqd;
	}

	public double sumResponses() {
		if (sum_responses_qty == 0){
			sum_responses_qty = 0.0;
			for (int i = 0; i < n_eta; i++){
				sum_responses_qty += responses[i];
			}
//			System.out.println("sum_responses_qty " + sum_responses_qty);
		}
		return sum_responses_qty;
	}	
		
	
	/**
	 * Gets the total number of predictors that could be used for rules at this juncture
	 * @return
	 */
	public int pAdj(){
		if (padj == null){
			padj = predictorsThatCouldBeUsedToSplitAtNode().size();
		}
		return padj;
	}
	
	protected ArrayList<Integer> predictorsThatCouldBeUsedToSplitAtNode() {
		if (possible_rule_variables == null){
			possible_rule_variables = new ArrayList<Integer>();
			
//			System.out.println("predictorsThatCouldBeUsedToSplitAtNode " + this.stringLocation(true) + " data is " + data.size() + " x " + data.get(0).length);
			
			for (int j = 0; j < cgmbart.getP(); j++){
				//if size of unique of x_i > 1
				double[] x_dot_j = Classifier.getColVector(data, j);
				//make hashset to get unique value
				TDoubleHashSet unique_x_dot_j = new TDoubleHashSet(x_dot_j);
				//now ensure that we have at least two unique vals to split on
				if (unique_x_dot_j.size() >= 2){
					possible_rule_variables.add(j);
				}
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
	
	protected TDoubleHashSet possibleSplitValuesGivenAttribute() {
		if (possible_split_vals_by_attr == null){
			possible_split_vals_by_attr = new HashMap<Integer, TDoubleHashSet>();
		}
		if (possible_split_vals_by_attr.get(splitAttributeM) == null){
			//super inefficient
			double[] x_dot_j = Classifier.getColVector(data, splitAttributeM);
			double max = Tools.max(x_dot_j);
			TDoubleHashSet unique_x_dot_j = new TDoubleHashSet(x_dot_j);			
			unique_x_dot_j.remove(max);
			possible_split_vals_by_attr.put(splitAttributeM, unique_x_dot_j);
		}
		return possible_split_vals_by_attr.get(splitAttributeM);
	}

	/**
	 * Pick a random predictor from the set of possible predictors at this juncture
	 * @return
	 */
	public int pickRandomPredictorThatCanBeAssigned(){
		ArrayList<Integer> predictors = predictorsThatCouldBeUsedToSplitAtNode();
		return predictors.get((int)Math.floor(StatToolbox.rand() * pAdj()));
	}
	
	public Double pickRandomSplitValue() {
		double[] split_values = possibleSplitValuesGivenAttribute().toArray();
		return split_values[(int) Math.floor(StatToolbox.rand() * split_values.length)];
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

	public int numTimesAttrUsed(int j) {
		if (this.isLeaf){
			return 0;
		}
		return (this.splitAttributeM == j ? 1 : 0) + this.left.numTimesAttrUsed(j) + this.right.numTimesAttrUsed(j);
	}

	public void setStumpData(ArrayList<double[]> X, double[] y, int p) {
		//pull out X data, set y's, and indices appropriately
		int n = X.size();
		
		responses = new double[n];
		indicies = new int[n];
		
		data = new ArrayList<double[]>(n);
		for (int i = 0; i < n; i++){
			indicies[i] = i;
			double[] x_i_dot = X.get(i);
			double[] x_i_dot_new = new double[p + 2];
			for (int j = 0; j < p + 2; j++){
				if (j == p){
					x_i_dot_new[j] = y[i];
					responses[i] = y[i];
				}
				else {
					x_i_dot_new[j] = x_i_dot[j];
				}
			}
			data.add(x_i_dot_new);
		}
		n_eta = data.size();
		
		//get
		
		//initialize the yhats
		yhats = new double[n];
//		System.out.println("setStumpData  X is " + data.size() + " x " + data.get(0).length + "  y is " + y.length + " x " + 1);
		printNodeDebugInfo("setStumpData");
	}

	public void printNodeDebugInfo(String title) {
		if (DEBUG_NODES){
			System.out.println("\n" + title + " node debug info for " + this.stringLocation(true) + (isLeaf ? " (LEAF) " : " (INTERNAL NODE) ") + " d = " + depth);
			System.out.println("-----------------------------------------");
			
			System.out.println("cgmbart = " + cgmbart + " parent = " + parent + " left = " + left + " right = " + right);
			System.out.println("----- RULE:   X_" + splitAttributeM + " <= " + splitValue + " ------");
			System.out.println("n_eta = " + n_eta + " y_pred = " + (y_pred == BAD_FLAG ? "BLANK" : cgmbart.un_transform_y_and_round(y_pred)));
			System.out.println("sum_responses_qty = " + sum_responses_qty + " sum_responses_qty_sqd = " + sum_responses_qty_sqd);
			
			System.out.println("possible_rule_variables: [" + Tools.StringJoin(possible_rule_variables, ", ") + "]");
			System.out.print("possible_split_vals_by_attr: {");
			if (possible_split_vals_by_attr != null){
				for (int key : possible_split_vals_by_attr.keySet()){
					System.out.print(" " + key + " -> [" + Tools.StringJoin(possible_split_vals_by_attr.get(key).toArray()) + "]");
				}
				System.out.print(" }\n");
			}
			else {
				System.out.println(" NULL MAP }");
			}
			
			System.out.println("X is " + data.size() + " x " + data.get(0).length + " and below:");
			for (int i = 0; i < data.size(); i++){
				double[] record = data.get(i).clone();
				record[cgmbart.p] = cgmbart.un_transform_y_and_round(record[cgmbart.p]);
				System.out.println("  " + Tools.StringJoin(record));
			}
			
			System.out.println("responses: (size " + responses.length + ") [" + Tools.StringJoin(cgmbart.un_transform_y_and_round(responses)) + "]");
			System.out.println("indicies: (size " + indicies.length + ") [" + Tools.StringJoin(indicies) + "]");
			if (Arrays.equals(yhats, new double[yhats.length])){
				System.out.println("y_hat_vec: (size " + yhats.length + ") [ BLANK ]");
			}
			else {
				System.out.println("y_hat_vec: (size " + yhats.length + ") [" + Tools.StringJoin(cgmbart.un_transform_y_and_round(yhats)) + "]");
			}
			System.out.println("-----------------------------------------\n\n\n");
		}
	}

	public void updateWithNewResponsesRecursively(double[] new_responses) {
		
//		System.out.println("updateWithNewResponsesRecursively " + this.stringLocation(true) + " indicies: " + Tools.StringJoin(indicies));
		//nuke previous responses and sums
		responses = new double[n_eta]; //ensure correct dimension
		sum_responses_qty_sqd = 0; //need to be primitives
		sum_responses_qty = 0; //need to be primitives
		//copy all the new data in appropriately
		for (int i = 0; i < n_eta; i++){
			double y_new = new_responses[indicies[i]];
			this.data.get(i)[cgmbart.p] = y_new;
			responses[i] = y_new;
		}
		if (DEBUG_NODES){
			System.out.println("new_responses: (size " + new_responses.length + ") [" + Tools.StringJoin(cgmbart.un_transform_y_and_round(new_responses)) + "]");
		}
		printNodeDebugInfo("updateWithNewResponsesRecursively");
		if (this.isLeaf){
			return;
		}		
		this.left.updateWithNewResponsesRecursively(new_responses);
		this.right.updateWithNewResponsesRecursively(new_responses);
	}

	public void updateYHatsWithPrediction() {		
		for (int i = 0; i < indicies.length; i++){
			yhats[indicies[i]] = y_pred;
		}
		printNodeDebugInfo("updateYHatsWithPrediction");
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