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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import TroveExtension.TDoubleHashSetAndArray;


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
	/** is this node a terminal leaf? */
	public boolean isLeaf;
	/** the attribute this node makes a decision on */
	public int splitAttributeM;
	/** the value this node makes a decision on */
	public double splitValue;
	/** if this is a leaf node, then the result of the classification, otherwise null */
	public Double klass;
	/** if this is a leaf node, then the result of the prediction for regression, otherwise null */
	protected static final double BAD_FLAG_double = -Double.MAX_VALUE;
	protected static final int BAD_FLAG_int = -Integer.MAX_VALUE;
	public double y_pred = BAD_FLAG_double;
	/** the remaining data records at this point in the tree construction cols: x_1, ..., x_p, y, index */
//	public transient List<double[]> data;
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
	private transient TIntArrayList possible_rule_variables;
	/** this caches the possible split values BY variable */
	private transient HashMap<Integer, TDoubleHashSetAndArray> possible_split_vals_by_attr;
	/** this caches the number of possible split variables */
	protected transient Integer padj;	
	
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
//		copy.data = data;
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
	
	public double avgResponse(){
		return StatToolbox.sample_average(responses);
	}
	
	//toolbox functions
	public ArrayList<CGMBARTTreeNode> getTerminalNodesWithDataAboveOrEqualToN(int n_rule){
		ArrayList<CGMBARTTreeNode> terminal_nodes_data_above_n = new ArrayList<CGMBARTTreeNode>();
		findTerminalNodesDataAboveOrEqualToN(this, terminal_nodes_data_above_n, n_rule);
		return terminal_nodes_data_above_n;
	}
	
	public ArrayList<CGMBARTTreeNode> getTerminalNodes(){
		return getTerminalNodesWithDataAboveOrEqualToN(0);
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
		node.splitAttributeM = BAD_FLAG_int;
		node.splitValue = BAD_FLAG_double;
	}

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
	
	public static final boolean DEBUG_NODES = false;
	public void propagateDataByChangedRule() {		
		if (isLeaf){ //only propagate if we are in a split node and NOT a leaf
			printNodeDebugInfo("propagateDataByChangedRule LEAF");
			return;
		}
		
		//split the data correctly
		TIntArrayList left_indices = new TIntArrayList(n_eta); 
		TIntArrayList right_indices = new TIntArrayList(n_eta);
		TDoubleArrayList left_responses = new TDoubleArrayList(n_eta);
		TDoubleArrayList right_responses = new TDoubleArrayList(n_eta);
		
		for (int i = 0; i < n_eta; i++){
			double[] datum = cgmbart.X_y.get(indicies[i]);
		
			if (datum[splitAttributeM] <= splitValue){
				left_indices.add(indicies[i]);
				left_responses.add(responses[i]);
			}
			else {
				right_indices.add(indicies[i]);
				right_responses.add(responses[i]);
			}
		}
		
		left.n_eta = left_responses.size();
		left.responses = left_responses.toArray();
		left.indicies = left_indices.toArray();
		
		right.n_eta = right_responses.size();
		right.responses = right_responses.toArray();
		right.indicies = right_indices.toArray();
		
		left.propagateDataByChangedRule();
		right.propagateDataByChangedRule();
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
			String split = this.splitAttributeM == BAD_FLAG_int ? "null" : TreeIllustration.two_digit_format.format(this.splitAttributeM);
			String value = this.splitValue == BAD_FLAG_double ? "null" : TreeIllustration.two_digit_format.format(this.splitValue);
			return "x_" + split + "  <  " + value;
		}
	}
	
	
	//CHECK as well
	public double prediction_untransformed(){
		if (y_pred == BAD_FLAG_double){
			return BAD_FLAG_double;
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
	
	protected TIntArrayList predictorsThatCouldBeUsedToSplitAtNode() {
		if (possible_rule_variables == null){
			possible_rule_variables = new TIntArrayList();
			
//			System.out.println("predictorsThatCouldBeUsedToSplitAtNode " + this.stringLocation(true) + " data is " + data.size() + " x " + data.get(0).length);
			
			for (int j = 0; j < cgmbart.p; j++){
				//if size of unique of x_i > 1
				double[] x_dot_j = cgmbart.X_y_by_col.get(j);
				
				for (int i = 1; i < indicies.length; i++){
					if (x_dot_j[indicies[i - 1]] != x_dot_j[indicies[i]]){
						possible_rule_variables.add(j);
						break;
					}
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
	
	protected TDoubleHashSetAndArray possibleSplitValuesGivenAttribute() {
		if (possible_split_vals_by_attr == null){
			possible_split_vals_by_attr = new HashMap<Integer, TDoubleHashSetAndArray>();
		}
		if (possible_split_vals_by_attr.get(splitAttributeM) == null){
			//super inefficient
			double[] x_dot_j = cgmbart.X_y_by_col.get(splitAttributeM);
			double[] x_dot_j_node = new double[n_eta];
			for (int i = 0; i < n_eta; i++){
				x_dot_j_node[i] = x_dot_j[indicies[i]];
			}
			
			TDoubleHashSetAndArray unique_x_dot_j_node = new TDoubleHashSetAndArray(x_dot_j_node);	
			double max = Tools.max(x_dot_j_node);
			unique_x_dot_j_node.remove(max);
			possible_split_vals_by_attr.put(splitAttributeM, unique_x_dot_j_node);
		}
		return possible_split_vals_by_attr.get(splitAttributeM);
	}


	public double pickRandomSplitValue() {	
		TDoubleHashSetAndArray split_values = possibleSplitValuesGivenAttribute();
//		if (splitAttributeM == 0){
//			double[] split_values_as_arr = split_values.getAsArray();
//			Arrays.sort(split_values_as_arr);
//			System.out.println("split_values: " + Tools.StringJoin(split_values_as_arr));
//		}
		int rand_index = (int) Math.floor(StatToolbox.rand() * split_values.size());
		return split_values.getAtIndex(rand_index);
	}
	
	public boolean isStump() {
		return parent == null && left == null && right == null;
	}
	public String stringID() {
		return toString().split("@")[1];
	}	
	
	
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

	public void setStumpData(ArrayList<double[]> X_y, double[] y_trans, int p) {
		//pull out X data, set y's, and indices appropriately
		n_eta = X_y.size();
		
		responses = new double[n_eta];
		indicies = new int[n_eta];
		
		
		for (int i = 0; i < n_eta; i++){
			indicies[i] = i;
		}
		for (int i = 0; i < n_eta; i++){
			for (int j = 0; j < p + 2; j++){
				if (j == p){
					responses[i] = y_trans[i];
				}
			}
		}	

		//initialize the yhats
		yhats = new double[n_eta];
//		System.out.println("setStumpData  X is " + data.size() + " x " + data.get(0).length + "  y is " + y.length + " x " + 1);
		printNodeDebugInfo("setStumpData");
	}

	public void printNodeDebugInfo(String title) {
		if (DEBUG_NODES){
			System.out.println("\n" + title + " node debug info for " + this.stringLocation(true) + (isLeaf ? " (LEAF) " : " (INTERNAL NODE) ") + " d = " + depth);
			System.out.println("-----------------------------------------");
			
			System.out.println("cgmbart = " + cgmbart + " parent = " + parent + " left = " + left + " right = " + right);
			System.out.println("----- RULE:   X_" + splitAttributeM + " <= " + splitValue + " ------");
			System.out.println("n_eta = " + n_eta + " y_pred = " + (y_pred == BAD_FLAG_double ? "BLANK" : cgmbart.un_transform_y_and_round(y_pred)));
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
			
//			System.out.println("X is " + data.size() + " x " + data.get(0).length + " and below:");
//			for (int i = 0; i < data.size(); i++){
//				double[] record = data.get(i).clone();
//				record[cgmbart.p] = cgmbart.un_transform_y_and_round(record[cgmbart.p]);
//				System.out.println("  " + Tools.StringJoin(record));
//			}
			
			System.out.println("responses: (size " + responses.length + ") [" + Tools.StringJoin(cgmbart.un_transform_y_and_round(responses)) + "]");
			System.out.println("indicies: (size " + indicies.length + ") [" + Tools.StringJoin(indicies) + "]");
			if (Arrays.equals(yhats, new double[yhats.length])){
				System.out.println("y_hat_vec: (size " + yhats.length + ") [ BLANK ]");
			}
			else {
				System.out.println("y_hat_vec: (size " + yhats.length + ") [" + Tools.StringJoin(cgmbart.un_transform_y_and_round(yhats)) + "]");
			}
			System.out.println("-----------------------------------------\n\n\n");
//			System.out.println("X_y y:   " + Tools.StringJoin(cgmbart.getResponses()));
//			System.out.println("y_trans: " + Tools.StringJoin(cgmbart.un_transform_y_and_round(cgmbart.y_trans)));
//			
//			System.out.println("-----------------------------------------\n\n\n");
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
//			this.data.get(i)[cgmbart.p] = y_new;
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

}