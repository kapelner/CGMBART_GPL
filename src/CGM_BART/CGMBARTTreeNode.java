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
import java.util.HashSet;

import OpenSourceExtensions.TDoubleHashSetAndArray;
import OpenSourceExtensions.UnorderedPair;



/**
 * A dumb struct to store information about a 
 * node in the Bayesian decision tree.
 * 
 * Unfortunately, this has parted ways too much with a CART
 * structure, so I needed to create something new
 * 
 * @author Adam Kapelner
 */
public class CGMBARTTreeNode implements Cloneable, Serializable {
	private static final long serialVersionUID = -5584590448078741112L;
	
	private static final int N_CUTOFF_CACHE = 2000;

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
	/** if this is a leaf node, then the result of the prediction for regression, otherwise null */
	protected static final double BAD_FLAG_double = -Double.MAX_VALUE;
	protected static final int BAD_FLAG_int = -Integer.MAX_VALUE;
	public double y_pred = BAD_FLAG_double;
	
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
	
	protected int[] attribute_split_counts;
	
	public CGMBARTTreeNode(){}	

	public CGMBARTTreeNode(CGMBARTTreeNode parent, CGMBART_02_hyperparams cgmbart){
		this.parent = parent;
		this.yhats = parent.yhats;
		this.attribute_split_counts = parent.attribute_split_counts;
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
//		//deep copy
//		if (possible_rule_variables != null){
//			TIntArrayList possible_rule_variables_clone = new TIntArrayList(possible_rule_variables.size());
//			possible_rule_variables_clone.addAll(possible_rule_variables);
//			copy.possible_rule_variables = possible_rule_variables_clone;
//		}
		copy.possible_rule_variables = possible_rule_variables;
		//deep copy
		copy.possible_split_vals_by_attr = possible_split_vals_by_attr;
		copy.depth = depth;
		//////do not copy y_pred
		//now do data stuff
//		copy.data = data;
		copy.responses = responses;
		copy.indicies = indicies;
		copy.n_eta = n_eta;
		copy.yhats = yhats;
		copy.attribute_split_counts = attribute_split_counts.clone();
		
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
		while (true){
			if (evalNode.isLeaf){
				return evalNode.y_pred;
			}
			//all split rules are less than or equals (this is merely a convention)
			//it's a convention that makes sense - if X_.j is binary, and the split values can only be 0/1
			//then it MUST be <= so both values can be considered
			if (xs[evalNode.splitAttributeM] <= evalNode.splitValue){
				evalNode = evalNode.left;
			}
			else {
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
			if (DEBUG_NODES){
				printNodeDebugInfo("propagateDataByChangedRule LEAF");
			}
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
	

	public int numNodesAndLeaves() {
		if (this.isLeaf){
			return 1;
		}
		else {
			return 1 + this.left.numNodesAndLeaves() + this.right.numNodesAndLeaves();
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
		else if (this.parent.right == this){
			return this.parent.stringLocation(false) + "R";
		}
		else {
			return this.parent.stringLocation(false) + "?";
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
		}
		return sum_responses_qty;
	}	
	
	protected TIntArrayList predictorsThatCouldBeUsedToSplitAtNode() {
		if (cgmbart.mem_cache_for_speed){
			if (possible_rule_variables == null){
				possible_rule_variables = tabulatePredictorsThatCouldBeUsedToSplitAtNode();
			}
			return possible_rule_variables;			
		}
		else {
			return tabulatePredictorsThatCouldBeUsedToSplitAtNode();
		}
	}
	
	private TIntArrayList tabulatePredictorsThatCouldBeUsedToSplitAtNode() {
		TIntArrayList possible_rule_variables = new TIntArrayList();
		
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
		if (cgmbart.mem_cache_for_speed){
			if (possible_split_vals_by_attr == null){
				possible_split_vals_by_attr = new HashMap<Integer, TDoubleHashSetAndArray>();
			}
			if (possible_split_vals_by_attr.get(splitAttributeM) == null){
				possible_split_vals_by_attr.put(splitAttributeM, tabulatePossibleSplitValuesGivenAttribute());
			}
			return possible_split_vals_by_attr.get(splitAttributeM);
		} 
		else {
			return tabulatePossibleSplitValuesGivenAttribute();
		}
	}
	
	private TDoubleHashSetAndArray tabulatePossibleSplitValuesGivenAttribute() {
		double[] x_dot_j = cgmbart.X_y_by_col.get(splitAttributeM);
		double[] x_dot_j_node = new double[n_eta];
		for (int i = 0; i < n_eta; i++){
			x_dot_j_node[i] = x_dot_j[indicies[i]];
		}
		
		TDoubleHashSetAndArray unique_x_dot_j_node = new TDoubleHashSetAndArray(x_dot_j_node);
		double max = Tools.max(x_dot_j_node);
		unique_x_dot_j_node.remove(max);
		return unique_x_dot_j_node;
	}

	public double pickRandomSplitValue() {	
		TDoubleHashSetAndArray split_values = possibleSplitValuesGivenAttribute();
		if (split_values.size() == 0){
			return CGMBARTTreeNode.BAD_FLAG_double;
		}
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

	public void numTimesAttrUsed(int[] total_for_trees) {
		if (this.isLeaf){
			return;
		}
		total_for_trees[this.splitAttributeM]++;
		left.numTimesAttrUsed(total_for_trees);
		right.numTimesAttrUsed(total_for_trees);
	}
	
	public void attrUsed(int[] total_for_trees) {
		if (this.isLeaf){
			return;
		}
		total_for_trees[this.splitAttributeM] = 1;
		left.numTimesAttrUsed(total_for_trees);
		right.numTimesAttrUsed(total_for_trees);
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
		//intialize the var counts
		attribute_split_counts = new int[p];
		
		if (DEBUG_NODES){printNodeDebugInfo("setStumpData");}
	}

	public void printNodeDebugInfo(String title) {		
		System.out.println("\n" + title + " node debug info for " + this.stringLocation(true) + (isLeaf ? " (LEAF) " : " (INTERNAL NODE) ") + " d = " + depth);
		System.out.println("-----------------------------------------");
		System.out.println("n_eta = " + n_eta + " y_pred = " + (y_pred == BAD_FLAG_double ? "BLANK" : cgmbart.un_transform_y_and_round(y_pred)));
		
		System.out.println("cgmbart = " + cgmbart + " parent = " + parent + " left = " + left + " right = " + right);
		
		if (this.parent != null){
			System.out.println("----- PARENT RULE:   X_" + parent.splitAttributeM + " <= " + parent.splitValue + " ------");
			//get vals of this x currently here
			double[] x_dot_j = cgmbart.X_y_by_col.get(parent.splitAttributeM);
			double[] x_dot_j_node = new double[n_eta];
			for (int i = 0; i < n_eta; i++){
				x_dot_j_node[i] = x_dot_j[indicies[i]];
			}
			Arrays.sort(x_dot_j_node);
			System.out.println("   all X_" + parent.splitAttributeM + " values here: [" + Tools.StringJoin(x_dot_j_node) + "]");
		}
		
		if (!isLeaf){
			System.out.println("\n----- RULE:   X_" + splitAttributeM + " <= " + splitValue + " ------");
			//get vals of this x currently here
			double[] x_dot_j = cgmbart.X_y_by_col.get(splitAttributeM);
			double[] x_dot_j_node = new double[n_eta];
			for (int i = 0; i < n_eta; i++){
				x_dot_j_node[i] = x_dot_j[indicies[i]];
			}
			Arrays.sort(x_dot_j_node);
			System.out.println("   all X_" + splitAttributeM + " values here: [" + Tools.StringJoin(x_dot_j_node) + "]");
		}	

		
		System.out.println("sum_responses_qty = " + sum_responses_qty + " sum_responses_qty_sqd = " + sum_responses_qty_sqd);
		
		System.out.println("possible_rule_variables: [" + Tools.StringJoin(possible_rule_variables, ", ") + "]");
		System.out.println("possible_split_vals_by_attr: {");
		if (possible_split_vals_by_attr != null){
			for (int key : possible_split_vals_by_attr.keySet()){
				double[] array = possible_split_vals_by_attr.get(key).toArray();
				Arrays.sort(array);
				System.out.println("  " + key + " -> [" + Tools.StringJoin(array) + "],");
			}
			System.out.print(" }\n");
		}
		else {
			System.out.println(" NULL MAP\n}");
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
//			System.out.println("X_y y:   " + Tools.StringJoin(cgmbart.getResponses()));
//			System.out.println("y_trans: " + Tools.StringJoin(cgmbart.un_transform_y_and_round(cgmbart.y_trans)));
//			
//			System.out.println("-----------------------------------------\n\n\n");
	}

	public void updateWithNewResponsesRecursively(double[] new_responses) {
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
			printNodeDebugInfo("updateWithNewResponsesRecursively");
		}
		
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
		if (DEBUG_NODES){
			printNodeDebugInfo("updateYHatsWithPrediction");
		}
	}

	public void findInteractions(HashSet<UnorderedPair<Integer>> set_of_interaction_pairs) {		
		if (this.isLeaf){
			return;
		}
		//add all pairs for which this split at this node interacts
		findSplitAttributesUsedUnderneath(this.splitAttributeM, set_of_interaction_pairs);
		//recurse further
		this.left.findInteractions(set_of_interaction_pairs);
		this.right.findInteractions(set_of_interaction_pairs);
		
	}

	private void findSplitAttributesUsedUnderneath(int interacted_attribute, HashSet<UnorderedPair<Integer>> set_of_interaction_pairs) {
		if (this.isLeaf){
			return;
		}
		//add new pair
		if (!this.left.isLeaf){
			set_of_interaction_pairs.add(new UnorderedPair<Integer>(interacted_attribute, this.left.splitAttributeM));
		}
		if (!this.right.isLeaf){
			set_of_interaction_pairs.add(new UnorderedPair<Integer>(interacted_attribute, this.right.splitAttributeM));
		}
		//now recurse
		this.left.findSplitAttributesUsedUnderneath(interacted_attribute, set_of_interaction_pairs);
		this.right.findSplitAttributesUsedUnderneath(interacted_attribute, set_of_interaction_pairs);
	}

	public void clearRulesAndSplitCache() {
		possible_rule_variables = null;
		possible_split_vals_by_attr = null;
	}

	public void decrement_variable_count(int j) {
		attribute_split_counts[j]--;
	}

	public void increment_variable_count(int j) {
		attribute_split_counts[j]++;
		
	}
	
//	public int sizeOfSplitVals(){
//		if (this.isLeaf){
//			return 0;
//		}
//		int sum = 0;
//		for (Integer key : possible_split_vals_by_attr.keySet()){
//			sum += possible_split_vals_by_attr.get(key).size();
//		}
//		return sum + this.left.sizeOfSplitVals() + this.right.sizeOfSplitVals();
//	}

}