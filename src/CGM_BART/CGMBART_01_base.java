package CGM_BART;

import java.io.Serializable;

public abstract class CGMBART_01_base extends Classifier implements Serializable {
	private static final long serialVersionUID = -7068937615952180038L;

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

	/** stuff for multi-threading and memory caching */
	protected int threadNum;
	protected int num_cores;
	protected boolean mem_cache_for_speed;
	
	/** should we print stuff out to screen? */
	protected boolean verbose = true;
	
	protected static Integer PrintOutEvery = null;
	

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
	
	public void setVerbose(boolean verbose){
		this.verbose = verbose;
	}
	
	public void setTotalNumThreads(int num_cores) {
		this.num_cores = num_cores;
	}	

	public void setMemCacheForSpeed(boolean mem_cache_for_speed){
		this.mem_cache_for_speed = mem_cache_for_speed;
	}
}
