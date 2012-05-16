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

public class CGMBARTPriorBuilder extends CGMTreePriorBuilder {
	
	public static double ALPHA = 0.95;
	public static double BETA = 2; //see p271 in CGM10
	
	protected double[] minimum_values_by_attribute;
	
	public CGMBARTPriorBuilder(ArrayList<double[]> X_y, int p) {
		super(X_y, p);
		//now let's go through and keep some more records
		recordMinAttrValues();
	}
	
	private void recordMinAttrValues() {
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
	
	public double getAlpha() {
		return ALPHA;
	}
	
	public double getBeta() {
		return BETA;
	}	
}
