package CGM_BART;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;



public abstract class CGMBART_base extends Classifier implements Serializable {
	private static final long serialVersionUID = -7068937615952180038L;


	protected static final int DEFAULT_NUM_TREES = 1;
	//this burn in number needs to be computed via some sort of moving average or time series calculation
	protected static final int DEFAULT_NUM_GIBBS_BURN_IN = 500;
	protected static final int DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS = 2000; //this must be larger than the number of burn in!!!
	
	protected static double ALPHA = 0.95;
	protected static double BETA = 2; //see p271 in CGM10	
	


	/** the actual list of trees */
	protected ArrayList<ArrayList<CGMBARTTreeNode>> gibbs_samples_of_cgm_trees;
	protected ArrayList<ArrayList<CGMBARTTreeNode>> gibbs_samples_of_cgm_trees_after_burn_in;
	/** the variance of the errors */
	protected ArrayList<Double> gibbs_samples_of_sigsq;
	protected ArrayList<Double> gibbs_samples_of_sigsq_after_burn_in;

	/** the current # of trees */
	protected int num_trees;

	/** useful metadata */
	protected double[] minimum_values_by_attribute;
	protected double[] maximum_values_by_attribute;
	//this is frequency of unique values by each column
	protected ArrayList<HashMap<Double, Integer>> num_val_hash_by_column;
	
	/** stuff during the build run time that we can access and look at */
	protected double[][] all_tree_liks;
	/** if the user pressed stop, we can cancel the Gibbs Sampling to unlock the CPU */
	protected boolean stop_bit;	
	protected static Integer PrintOutEvery = null;
	
	
	
	public CGMBART_base() {
		super();
//		System.out.println("CGMBART constructor");
		num_trees = DEFAULT_NUM_TREES;
	}	
	
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		
		//set useful metadata
		recordMinAndMaxAttributeValues();
		recordHashesForNumValues();	
	}

	public void setNumTrees(int m){
		this.num_trees = m;
	}
	
	public void setPrintOutEveryNIter(int print_out_every){
		PrintOutEvery = print_out_every;
	}
	
	
	public void setAlpha(double alpha){
		ALPHA = alpha;
	}
	
	public void setBeta(double beta){
		BETA = beta;
	}	
	
	public double getAlpha(){
		return ALPHA;
	}
	
	public double getBeta(){
		return BETA;
	}	

	
	protected void recordHashesForNumValues() {
		num_val_hash_by_column = new ArrayList<HashMap<Double, Integer>>();
		for (int j = 0; j < p; j++){
			HashMap<Double, Integer> num_val_hash = new HashMap<Double, Integer>();
			for (int i = 0; i < n; i++){
				double val = X_y.get(i)[j];
				Integer freq = num_val_hash.get(val);
				if (freq == null){
					num_val_hash.put(val, 1);
				}
				else {
					num_val_hash.put(val, freq + 1);					
				}
			}
			num_val_hash_by_column.add(num_val_hash);
		}		
	}
	
	public int frequencyValueForAttribute(int attribute, double val){
		HashMap<Double, Integer> attr_vals = num_val_hash_by_column.get(attribute);
		if (attr_vals.get(val) == null){
			System.out.println("attr_vals.get(val) == null");
			System.out.println(Tools.StringJoin(attr_vals.keySet().toArray(), ","));
		}
		return attr_vals.get(val);
	}

	protected void recordMinAndMaxAttributeValues() {
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
		
		maximum_values_by_attribute = new double[p];
		for (int j = 0; j < p; j++){
			double max = Double.MIN_VALUE;
			for (int i = 0; i < n; i++){
				if (X_y.get(i)[j] > max){
					max = X_y.get(i)[j];
				}
			}
			maximum_values_by_attribute[j] = max;
		}
	}	
	
	public double[] getMinimum_values_by_attribute() {
		return minimum_values_by_attribute;
	}
	public double[] getMaximum_values_by_attribute() {
		return maximum_values_by_attribute;
	}	


	@Override
	protected void FlushData() {
		for (ArrayList<CGMBARTTreeNode> cgm_trees : gibbs_samples_of_cgm_trees){
			FlushDataForSample(cgm_trees);
		}	
	}
	

	protected void FlushDataForSample(ArrayList<CGMBARTTreeNode> cgm_trees) {
		for (CGMBARTTreeNode tree : cgm_trees){
			tree.flushNodeData();	
		}
	}	

	@Override
	public void StopBuilding() {
		stop_bit = true;
	}	

}
