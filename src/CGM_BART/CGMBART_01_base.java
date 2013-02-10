package CGM_BART;

import java.io.Serializable;

public abstract class CGMBART_01_base extends Classifier implements Serializable {
	private static final long serialVersionUID = -7068937615952180038L;

//	protected static final int DEFAULT_NUM_TREES = 1;
//	//this burn in number needs to be computed via some sort of moving average or time series calculation
//	protected static final int DEFAULT_NUM_GIBBS_BURN_IN = 10;
//	protected static final int DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS = 20; //this must be larger than the number of burn in!!!

	protected static final int DEFAULT_NUM_TREES = 200;
	//this burn in number needs to be computed via some sort of moving average or time series calculation
	protected static final int DEFAULT_NUM_GIBBS_BURN_IN = 1000;
	protected static final int DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS = 2000; //this must be larger than the number of burn in!!!
	
	
	/** the actual list of trees */
	protected CGMBARTTreeNode[][] gibbs_samples_of_cgm_trees;
	protected CGMBARTTreeNode[][] gibbs_samples_of_cgm_trees_after_burn_in;
	/** the variance of the errors */
	protected double[] gibbs_samples_of_sigsq;
	protected double[] gibbs_samples_of_sigsq_after_burn_in;
	/** accept or reject the MH step */
	protected boolean[][] accept_reject_mh;
	protected char[][] accept_reject_mh_steps;

	/** the current # of trees */
	protected int num_trees;
	protected int num_gibbs_burn_in;
	protected int num_gibbs_total_iterations;
	
	/** if the user pressed stop, we can cancel the Gibbs Sampling to unlock the CPU */
	protected boolean stop_bit;

	protected int threadNum;
	
	protected static Integer PrintOutEvery = null;
	
	
	
	public CGMBART_01_base() {
		super();
//		System.out.println("CGMBART constructor");
		num_trees = DEFAULT_NUM_TREES;
		num_gibbs_burn_in = DEFAULT_NUM_GIBBS_BURN_IN;
		num_gibbs_total_iterations = DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS;		
	}

	public void setNumTrees(int m){
//		System.out.println("set num trees = " + num_trees);
		this.num_trees = m;
	}
	
	public void setPrintOutEveryNIter(int print_out_every){
		PrintOutEvery = print_out_every;
	}

	@Override
	protected void FlushData() {
		for (CGMBARTTreeNode[] cgm_trees : gibbs_samples_of_cgm_trees){
			FlushDataForSample(cgm_trees);
		}	
	}
	

	protected void FlushDataForSample(CGMBARTTreeNode[] cgm_trees) {
		for (CGMBARTTreeNode tree : cgm_trees){
			tree.flushNodeData();	
		}
	}	

	@Override
	public void StopBuilding() {
		stop_bit = true;
	}	
	
	public long maxMemory(){
		return Runtime.getRuntime().maxMemory();
	}


	public void setThreadNum(int threadNum) {
		this.threadNum = threadNum;
	}
}
