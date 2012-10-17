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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


public abstract class ClassificationAndRegressionTree extends Classifier {
	private static final long serialVersionUID = 3362976822356958024L;

	/** This is the root of the Decision Tree, the only thing that is 
	 * saved during serialization, you must override this field! */
	protected CGMBARTTreeNode root;

	public ClassificationAndRegressionTree(){}
	
	/** deletes all records from the tree. Usually a wrapper for a recursive function */
	public abstract void FlushData();

	public void setRoot(CGMBARTTreeNode root) {
		this.root = root;
	}
	
	//convenience methods for all tree building...
	
	/**
	 * Sorts a data matrix by an attribute from lowest record to highest record
	 * 
	 * @param data			the data matrix to be sorted
	 * @param j				the attribute to sort on
	 */
	@SuppressWarnings("unchecked")
	public static void SortAtAttribute(List<double[]> data, int j){
		Collections.sort(data, new AttributeComparator(j));
	}
	
	
	/**
	 * This class compares two data records by numerically comparing a specified attribute
	 * 
	 * @author Adam Kapelner
	 */
	@SuppressWarnings("rawtypes")
	private static class AttributeComparator implements Comparator {
		
		/** the specified attribute */
		private int j;
		/**
		 * Create a new comparator
		 * @param j			the attribute in which to compare on
		 */
		public AttributeComparator(int j){
			this.j = j;
		}
		/**
		 * Compare the two data records. They must be of type int[].
		 * 
		 * @param o1		data record A
		 * @param o2		data record B
		 * @return			-1 if A[m] < B[m], 1 if A[m] > B[m], 0 if equal
		 */
		public int compare(Object o1, Object o2){
			double a = ((double[])o1)[j];
			double b = ((double[])o2)[j];
			if (a < b)
				return -1;
			if (a > b)
				return 1;
			else
				return 0;
		}		
	}
	
	public static int getSplitPoint(List<double[]> data, int splitAttribute, double splitValue){
		//if we started with no data, the split point should just be the zero index for consistency
		if (data.isEmpty()){
			return 0;
		}
		for (int i = 0; i < data.size(); i++){
			if (data.get(i)[splitAttribute] > splitValue){
				return i;
			}
		}
		return data.size() - 1; //all the data is less than this split point
	}
	
	/**
	 * Split a data matrix and return the upper portion
	 * 
	 * @param data		the data matrix to be split
	 * @param nSplit	return all data records above this index in a sub-data matrix
	 * @param p 
	 * @param node_left_indicies 
	 * @param node_left_responses 
	 * @return			the upper sub-data matrix
	 */
	public static List<double[]> getUpperPortion(List<double[]> data, int nSplit, double[] responses, int[] indicies, int p){
		int N = data.size();
		List<double[]> upper = new ArrayList<double[]>(N - nSplit);
		for (int n = nSplit; n < N; n++){
			double[] record = data.get(n);
			upper.add(record);
			responses[n - nSplit] = record[p];
			indicies[n - nSplit] = (int)record[p + 1];			
		}
		return upper;
	}
	
	/**
	 * Split a data matrix and return the lower portion
	 * 
	 * @param data		the data matrix to be split
	 * @param nSplit	return all data records equal to or below this index in a sub-data matrix
	 * @param node_left_indicies 
	 * @param node_left_responses 
	 * @param p 
	 * @return			the lower sub-data matrix
	 */
	public static List<double[]> getLowerPortion(List<double[]> data, int nSplit, double[] responses, int[] indicies, int p){
		List<double[]> lower = new ArrayList<double[]>(nSplit);
		for (int n = 0; n < nSplit; n++){
			double[] record = data.get(n);
			lower.add(record);
			responses[n] = record[p];
			indicies[n] = (int)record[p + 1];
		}
		return lower;
	}
}
