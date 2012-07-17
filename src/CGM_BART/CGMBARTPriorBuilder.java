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

import CGM_Statistics.CGMTreeNode;
import CGM_Statistics.CGMTreePriorBuilder;
import CGM_Statistics.ClassificationAndRegressionTree;
import CGM_Statistics.StatToolbox;

public class CGMBARTPriorBuilder {
	
	public static double ALPHA = 0.95;
	public static double BETA = 2; //see p271 in CGM10
	
	protected ArrayList<double[]> X_y;
	protected int p;
	protected int n;	
	protected double[] minimum_values_by_attribute;
	
	public CGMBARTPriorBuilder(ArrayList<double[]> X_y) {
		this.X_y = X_y;
		this.p = X_y.get(0).length - 1;
		this.n = X_y.size();
		//now let's go through and keep some more records
		recordMinimumAttributeValues();
	}
	
	private void recordMinimumAttributeValues() {
		minimum_values_by_attribute = new double[p];
		for (int j = 0; j < p; j++){
			double min = Double.MAX_VALUE;
			for (int i = 0; i < n; i++){
				if (X_y.get(i)[j] < min){
					min = X_y.get(i)[j];
				}
			}
			minimum_values_by_attribute[j] = min;
		}
	}
	
	public ArrayList<Integer> predictorsThatCouldBeUsedToSplitAtNode(CGMTreeNode node){
		ArrayList<Integer> predictors = new ArrayList<Integer>();
		for (int j = 0; j < p; j++){
			//okay we can only add a predictor here if we don't see the minimum 
			//value in any of the nodes above split rules
			boolean can_use = true;
			System.out.println("predictorsThatCouldBeUsedToSplitAtNode" + node);
			for (CGMTreeNode father : node.getLineage()){
				if (father.splitAttributeM == j && father.splitValue == minimum_values_by_attribute[j]){
					can_use = false;
					break;
				}
			}
			if (can_use){
				predictors.add(j);
			}
		}		
		return predictors;
	}
	
	public ArrayList<Double> possibleSplitValues(CGMTreeNode node) {
		//we need to look above in the lineage and get the minimum value that was previously split on
		ArrayList<Double> previous_split_points = new ArrayList<Double>();
		for (CGMTreeNode father : node.getLineage()){
			if (father.splitAttributeM == node.splitAttributeM){
				previous_split_points.add(father.splitValue);				
			}
		}
		
		double min_split_value = Double.MAX_VALUE;
		for (int i = 0; i < previous_split_points.size(); i++){
			if (previous_split_points.get(i) < min_split_value){
				min_split_value = previous_split_points.get(i);
			}
		}
		
		//now we need to look in the design matrix and see what values are available
		ArrayList<Double> possible_values = new ArrayList<Double>();
		for (int i = 0; i < n; i++){
			if (X_y.get(i)[node.splitAttributeM] < min_split_value){
				possible_values.add(X_y.get(i)[node.splitAttributeM]);
			}
		}	
		return possible_values;
	}
	
	public double assignSplitValue(CGMTreeNode node) {
		return node.possible_split_values.get((int) Math.floor(StatToolbox.rand() * node.possible_split_values.size()));
	}	

	public Integer assignSplitAttribute(CGMTreeNode node) {
		ArrayList<Integer> could_be_used = predictorsThatCouldBeUsedToSplitAtNode(node);
		return could_be_used.get((int) Math.floor(StatToolbox.rand() * could_be_used.size()));
	}	
	
	public boolean splitNodeAndAssignRule(CGMTreeNode node) {		
		//first assign a split attribute
		
		node.splitAttributeM = assignSplitAttribute(node);
		//if 
		//a) we don't have any attributes left to split on, 
		//b) there's only one data point left
		//then this node automatically becomes a leaf, otherwise split it
		if (node.splitAttributeM == null || node.data.size() == 1){
			node.isLeaf = true;
			return false; //we did not do a split
		}	
		else {
			//we're no longer a leaf if we once were
			node.isLeaf = false;
			node.klass = null;
			//assign a splitting value
			node.splitValue = assignSplitValue(node);			
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
	
	public int getP() {
		return p;
	}	
	
	public double[] getMinimumValuesByAttribute(){
		return minimum_values_by_attribute;
	}
	
	public double getAlpha() {
		return ALPHA;
	}
	
	public double getBeta() {
		return BETA;
	}	
}
