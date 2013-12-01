package CGM_BART;

import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import OpenSourceExtensions.UnorderedPair;

/**
 * This class handles the parallelization of many Gibbs chains over many CPU cores
 * to create one BART regression model. It also handles all operations on the completed model.
 * 
 * @author Adam Kapelner and Justin Bleich
 */
public class CGMBARTRegressionMultThread extends Classifier {
	
	/** the number of CPU cores to build many different Gibbs chain within a BART model */
	protected int num_cores = 1;
	/** the number of trees in this BART model on all Gibbs chains */
	protected int num_trees = 50;
	
	/** the collection of <code>num_cores</code> BART models which will run separate Gibbs chains */
	protected transient CGMBARTRegression[] bart_gibbs_chain_threads;
	/** this is the combined gibbs samples after burn in from all of the <code>num_cores</code> chains */
	protected CGMBARTTreeNode[][] gibbs_samples_of_cgm_trees_after_burn_in;
	
	/** the estimate of some upper limit of the variance of the response which is usually the MSE from a a linear regression */
	private Double sample_var_y;
	/** the number of burn-in samples in each Gibbs chain */
	protected int num_gibbs_burn_in = 250;
	/** the total number of gibbs samples where each chain gets a number of burn-in and then the difference from the total divided by <code>num_cores</code> */ 
	protected int num_gibbs_total_iterations = 1250;
	/** the total number of Gibbs samples for each of the <code>num_cores</code> chains */
	protected int total_iterations_multithreaded;

	/** The probability vector that samples covariates for selecting split rules */
	protected double[] cov_split_prior;
	/** A hyperparameter that controls how easy it is to grow new nodes in a tree independent of depth */
	protected Double alpha = 0.95;
	/** A hyperparameter that controls how easy it is to grow new nodes in a tree dependent on depth which makes it more difficult as the tree gets deeper */
	protected Double beta = 2.0;
	/** this controls where to set <code>hyper_sigsq_mu</code> by forcing the variance to be this number of standard deviations on the normal CDF */
	protected Double hyper_k = 2.0;
	/** At a fixed <code>hyper_nu</code>, this controls where to set <code>hyper_lambda</code> by forcing q proportion to be at that value in the inverse gamma CDF */
	protected Double hyper_q = 0.9;
	/** half the shape parameter and half the multiplicand of the scale parameter of the inverse gamma prior on the variance */
	protected Double hyper_nu = 3.0;
	/** the hyperparameter of the probability of picking a grow step during the Metropolis-Hastings tree proposal */
	protected Double prob_grow = 2.5 / 9.0;
	/** the hyperparameter of the probability of picking a prune step during the Metropolis-Hastings tree proposal */
	protected Double prob_prune = 2.5 / 9.0;
	
	/** should we print select messages to the screen */
	protected boolean verbose = true;
	/** 
	 * whether or not we use the memory cache feature
	 * 
	 * @see Section 3.1 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
	 */
	protected boolean mem_cache_for_speed = true;
	/** "Destroyed" means this model's Gibbs samplers' data has been released to RAM and hence cannot be operated on */
	protected boolean destroyed;

	
	/** the default constructor sets the number of total iterations each Gibbs chain is charged with sampling */
	public CGMBARTRegressionMultThread(){	
		setNumGibbsTotalIterations(num_gibbs_total_iterations);
	}
	
	/**
	 * This is a simple setter for the number of total Gibbs samples and
	 * it also sets the total iterations per Gibbs chain running on each CPU
	 * core (see formula in the code)
	 * 
	 * @param num_gibbs_total_iterations	The number of total Gibbs iterations to set
	 */
	public void setNumGibbsTotalIterations(int num_gibbs_total_iterations){
		this.num_gibbs_total_iterations = num_gibbs_total_iterations;
		total_iterations_multithreaded = num_gibbs_burn_in + (int)Math.ceil((num_gibbs_total_iterations - num_gibbs_burn_in) / (double) num_cores);
	}
	
	/** The number of samples after burning is simply the title minus the burn-in */
	public int numSamplesAfterBurning(){
		return num_gibbs_total_iterations - num_gibbs_burn_in;
	}
	
	/** Set up an array of regression BARTs with length equal to <code>num_cores</code>, the number of CPU cores requested */
	protected void SetupBARTModels() {
		bart_gibbs_chain_threads = new CGMBARTRegression[num_cores];
		for (int t = 0; t < num_cores; t++){
			CGMBARTRegression bart = new CGMBARTRegression();
			SetupBartModel(bart, t);
		}
	}

	/**
	 * Initialize one of the <code>num_cores</code> BART models by setting
	 * all its custom parameters
	 * 
	 * @param bart		The BART model to initialize
	 * @param t			The number of the core this BART model corresponds to
	 */
	protected void SetupBartModel(CGMBARTRegression bart, int t) {
		bart.setVerbose(verbose);
		//now set specs on each of the bart models
		bart.num_trees = num_trees;
		bart.num_gibbs_total_iterations = total_iterations_multithreaded;
		bart.num_gibbs_burn_in = num_gibbs_burn_in;
		bart.sample_var_y = sample_var_y;		
		//now some hyperparams
		bart.setAlpha(alpha);
		bart.setBeta(beta);
		bart.setK(hyper_k);
		bart.setProbGrow(prob_grow);
		bart.setProbPrune(prob_prune);
		//set thread num and data
		bart.setThreadNum(t);
		bart.setTotalNumThreads(num_cores);
		bart.setMemCacheForSpeed(mem_cache_for_speed);
		
		//set features
		if (cov_split_prior != null){
			bart.setCovSplitPrior(cov_split_prior);
		}
		//do special stuff for regression model
		if (!(bart instanceof CGMBARTClassification)){
			bart.setNu(hyper_nu);		
			bart.setQ(hyper_q);
		}
		//once the params are set, now you can set the data
		bart.setData(X_y);
		bart_gibbs_chain_threads[t] = bart;
	}
	
	/**
	 * Takes a library of standard normal samples provided external and caches them
	 * 
	 * @param norm_samples	The externally provided cache
	 */
	public void setNormSamples(double[] norm_samples){
		CGMBART_02_hyperparams.samps_std_normal = norm_samples;
		CGMBART_02_hyperparams.samps_std_normal_length = norm_samples.length;
	}
	
	/**
	 * Takes a library of chi-squared samples provided external and caches them
	 * 
	 * @param norm_samples	The externally provided cache
	 */
	public void setGammaSamples(double[] gamma_samples){
		CGMBART_02_hyperparams.samps_chi_sq_df_eq_nu_plus_n = gamma_samples;
		CGMBART_02_hyperparams.samps_chi_sq_df_eq_nu_plus_n_length = gamma_samples.length;
	}

	/** This function actually initiates the Gibbs sampling to build all the BART models */
	public void Build() {
		SetupBARTModels();
		//run a build on all threads
		long t0 = System.currentTimeMillis();
		if (verbose){
			System.out.println("building BART " + (mem_cache_for_speed ? "with" : "without") + " mem-cache speedup");
		}
		BuildOnAllThreads();
		long t1 = System.currentTimeMillis();
		if (verbose){
			System.out.println("done building BART in " + ((t1 - t0) / 1000.0) + " sec \n");
		}
		//once it's done, now put together the chains
		ConstructBurnedChainForTreesAndOtherInformation();
	}	
	
	/** Create a post burn-in chain for ease of manipulation later */
	protected void ConstructBurnedChainForTreesAndOtherInformation() {
		gibbs_samples_of_cgm_trees_after_burn_in = new CGMBARTTreeNode[numSamplesAfterBurning()][num_trees];

		if (verbose){
			System.out.print("burning and aggregating chains from all threads... ");
		}
		//go through each thread and get the tail and put them together
		for (int t = 0; t < num_cores; t++){
			CGMBARTRegression bart_model = bart_gibbs_chain_threads[t];
			for (int i = num_gibbs_burn_in; i < total_iterations_multithreaded; i++){
				int offset = t * (total_iterations_multithreaded - num_gibbs_burn_in);
				int g = offset + (i - num_gibbs_burn_in);
//				System.out.println("t = " + t + " total_iterations_multithreaded = " + total_iterations_multithreaded + " g = " + g);
				if (g >= numSamplesAfterBurning()){
					break;
				}
				gibbs_samples_of_cgm_trees_after_burn_in[g] = bart_model.gibbs_samples_of_cgm_trees[i];
			}			
		}
		if (verbose){
			System.out.print("done\n");
		}
	}

	/** This is the core of BART's parallelization for model creation: build one BART model on each CPU core in parallel */
	private void BuildOnAllThreads(){
		ExecutorService bart_gibbs_chain_pool = Executors.newFixedThreadPool(num_cores);
		for (int t = 0; t < num_cores; t++){
			final int tf = t;
	    	bart_gibbs_chain_pool.execute(new Runnable(){
				public void run() {
					bart_gibbs_chain_threads[tf].Build();
				}
			});
		}
		bart_gibbs_chain_pool.shutdown();
		try {	         
	         bart_gibbs_chain_pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS); //effectively infinity
	    } catch (InterruptedException ignored){}		
	}

	/**
	 * Return the predictions from each tree for each burned-in Gibbs sample
	 * 
	 * 
	 * .
	 * 
	 * @param records
	 * @param num_cores_evaluate02
	 * @return
	 */
	protected double[][] getGibbsSamplesForPrediction(final double[][] records, final int num_cores_evaluate){
		final int num_samples_after_burn_in = numSamplesAfterBurning();
		final CGMBARTRegression first_bart = bart_gibbs_chain_threads[0];
		
		final int n = records.length;
		final double[][] y_hat = new double[n][records[0].length];
		
		//this is really ugly, but it's faster (we've checked in a Profiler)
		if (num_cores_evaluate == 1){
			for (int i = 0; i < n; i++){
				double[] y_gibbs_samples = new double[num_samples_after_burn_in];
				for (int g = 0; g < num_samples_after_burn_in; g++){
					CGMBARTTreeNode[] cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in[g];
					double yt_i = 0;
					for (int m = 0; m < num_trees; m++){ //sum of trees right?
						yt_i += cgm_trees[m].Evaluate(records[i]);
					}
					//just make sure we switch it back to really what y is for the user
					y_gibbs_samples[g] = first_bart.un_transform_y(yt_i);
				}
				y_hat[i] = y_gibbs_samples;
			}			
		}
		else {
			Thread[] fixed_thread_pool = new Thread[num_cores_evaluate];
			for (int t = 0; t < num_cores_evaluate; t++){
				final int final_t = t;
				Thread thread = new Thread(){
					public void run(){
						for (int i = 0; i < n; i++){
							if (i % num_cores_evaluate == final_t){
								double[] y_gibbs_samples = new double[num_samples_after_burn_in];
								for (int g = 0; g < num_samples_after_burn_in; g++){									
									CGMBARTTreeNode[] trees = gibbs_samples_of_cgm_trees_after_burn_in[g];
									double yt_i = 0;
									for (int m = 0; m < num_trees; m++){ //sum of trees right?
										yt_i += trees[m].Evaluate(records[i]);
									}
									//just make sure we switch it back to really what y is for the user
									y_gibbs_samples[g] = first_bart.un_transform_y(yt_i);	
								}
								y_hat[i] = y_gibbs_samples;
							}
						}
					}
				};
				thread.start();
				fixed_thread_pool[t] = thread;
			}
			for (int t = 0; t < num_cores_evaluate; t++){
				try {
					fixed_thread_pool[t].join();
				} catch (InterruptedException e) {}
			}
		}		

		return y_hat;
	}

	protected double[] getPostPredictiveIntervalForPrediction(double[] record, double coverage, int num_cores_evaluate){
		double[][] data = new double[1][record.length];
		data[0] = record;		
		//get all gibbs samples sorted
		double[][] y_gibbs_samples_sorted_matrix = getGibbsSamplesForPrediction(data, num_cores_evaluate);
		double[] y_gibbs_samples_sorted = y_gibbs_samples_sorted_matrix[0];
		Arrays.sort(y_gibbs_samples_sorted);
		
		//calculate index of the CI_a and CI_b
		int n_bottom = (int)Math.round((1 - coverage) / 2 * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
		int n_top = (int)Math.round(((1 - coverage) / 2 + coverage) * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
		double[] conf_interval = {y_gibbs_samples_sorted[n_bottom], y_gibbs_samples_sorted[n_top]};
		return conf_interval;
	}
	
	protected double[] get95PctPostPredictiveIntervalForPrediction(double[] record, int num_cores_evaluate){
		return getPostPredictiveIntervalForPrediction(record, 0.95, num_cores_evaluate);
	}	
	
	public double[] getGibbsSamplesSigsqs(){
		TDoubleArrayList sigsqs_to_export = new TDoubleArrayList(num_gibbs_total_iterations);
		for (int t = 0; t < num_cores; t++){
			TDoubleArrayList sigsqs_to_export_by_thread = new TDoubleArrayList(bart_gibbs_chain_threads[t].getGibbsSamplesSigsqs());
			if (t == 0){
				sigsqs_to_export.addAll(sigsqs_to_export_by_thread);
			}
			else {
				sigsqs_to_export.addAll(sigsqs_to_export_by_thread.subList(num_gibbs_burn_in, total_iterations_multithreaded));
			}
		}
		return sigsqs_to_export.toArray();
	}
	
	public boolean[][] getAcceptRejectMHsBurnin(){
		boolean[][] accept_reject_mh_first_thread = bart_gibbs_chain_threads[0].getAcceptRejectMH();
		boolean[][] accept_reject_mh_burn_ins = new boolean[num_gibbs_burn_in][num_trees];
		for (int g = 1; g < num_gibbs_burn_in + 1; g++){
			accept_reject_mh_burn_ins[g - 1] = accept_reject_mh_first_thread[g];
		}
		return accept_reject_mh_burn_ins;
	}
	
	public boolean[][] getAcceptRejectMHsAfterBurnIn(int thread_num){
		boolean[][] accept_reject_mh_by_core = bart_gibbs_chain_threads[thread_num - 1].getAcceptRejectMH();
		boolean[][] accept_reject_mh_after_burn_ins = new boolean[total_iterations_multithreaded - num_gibbs_burn_in][num_trees];
		for (int g = num_gibbs_burn_in; g < total_iterations_multithreaded; g++){
			accept_reject_mh_after_burn_ins[g - num_gibbs_burn_in] = accept_reject_mh_by_core[g];
		}
		return accept_reject_mh_after_burn_ins;
	}	
	
	public double[] getAttributeProps(final int num_cores, final String type) {
		int[][] variable_counts_all_gibbs = getCountsForAllAttribute(num_cores, type);
		double[] attribute_counts = new double[p];
		for (int g = 0; g < num_gibbs_total_iterations - num_gibbs_burn_in; g++){
			attribute_counts = Tools.add_arrays(attribute_counts, variable_counts_all_gibbs[g]);
		}
		Tools.normalize_array(attribute_counts); //will turn it into proportions
		return attribute_counts;
	}

	public int[][] getCountsForAllAttribute(final int num_cores, final String type) {
		final int[][] variable_counts_all_gibbs = new int[num_gibbs_total_iterations - num_gibbs_burn_in][p];		
		
		for (int g = 0; g < num_gibbs_total_iterations - num_gibbs_burn_in; g++){
			final CGMBARTTreeNode[] trees = gibbs_samples_of_cgm_trees_after_burn_in[g];
			int[] variable_counts_one_gibbs = new int[p];
			for (CGMBARTTreeNode tree : trees){	
				if (type.equals("splits")){
					variable_counts_one_gibbs = Tools.add_arrays(variable_counts_one_gibbs, tree.attribute_split_counts);
				}
				else if (type.equals("trees")){
					variable_counts_one_gibbs = Tools.binary_add_arrays(variable_counts_one_gibbs, tree.attribute_split_counts);
				}				
				
			}
			variable_counts_all_gibbs[g] = variable_counts_one_gibbs;
		}		
		return variable_counts_all_gibbs;
	}

	/** Flush all unnecessary data from the Gibbs chains to conserve RAM */
	protected void FlushData() {
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].FlushData();
		}
	}

	public double Evaluate(double[] record) {	
		return EvaluateViaSampAvg(record, 1);
	}	
	
	public double Evaluate(double[] record, int num_cores_evaluate) {		
		return EvaluateViaSampAvg(record, num_cores_evaluate);
	}		
	
	public double EvaluateViaSampMed(double[] record, int num_cores_evaluate) {	
		double[][] data = new double[1][record.length];
		data[0] = record;
		double[][] gibbs_samples = getGibbsSamplesForPrediction(data, num_cores_evaluate);
		return StatToolbox.sample_median(gibbs_samples[0]);
	}
	
	public double EvaluateViaSampAvg(double[] record, int num_cores_evaluate) {		
		double[][] data = new double[1][record.length];
		data[0] = record;
		double[][] gibbs_samples = getGibbsSamplesForPrediction(data, num_cores_evaluate);
		return StatToolbox.sample_average(gibbs_samples[0]);
	}

	public double[] getSigsqsByGibbsSample(int g){
		return bart_gibbs_chain_threads[0].un_transform_sigsq(bart_gibbs_chain_threads[0].gibbs_samples_of_sigsq);
	}	
		
	public int[][] getDepthsForTreesInGibbsSampAfterBurnIn(int thread_num){
		return bart_gibbs_chain_threads[thread_num - 1].getDepthsForTrees(num_gibbs_burn_in, total_iterations_multithreaded);
	}	
	
	public int[][] getNumNodesAndLeavesForTreesInGibbsSampAfterBurnIn(int thread_num){
		return bart_gibbs_chain_threads[thread_num - 1].getNumNodesAndLeavesForTrees(num_gibbs_burn_in, total_iterations_multithreaded);
	}	
	
	public void destroy(){
		bart_gibbs_chain_threads = null;
		gibbs_samples_of_cgm_trees_after_burn_in = null;
		cov_split_prior = null;
		destroyed = true; 
	}
	

	public int[][] getInteractionCounts(int num_cores){
		int[][] interaction_count_matrix = new int[p][p];
		
		for (int g = 0; g < gibbs_samples_of_cgm_trees_after_burn_in.length; g++){
			CGMBARTTreeNode[] trees = gibbs_samples_of_cgm_trees_after_burn_in[g];
			
			for (CGMBARTTreeNode tree : trees){
				//get the set of pairs of interactions
				HashSet<UnorderedPair<Integer>> set_of_interaction_pairs = new HashSet<UnorderedPair<Integer>>(p * p);
				//find all interactions
				tree.findInteractions(set_of_interaction_pairs);
				//now tabulate these interactions in our count matrix
				for (UnorderedPair<Integer> pair : set_of_interaction_pairs){
					interaction_count_matrix[pair.getFirst()][pair.getSecond()]++; 
				}
			}	
		}
		
		return interaction_count_matrix;
	}
	
	public void setData(ArrayList<double[]> X_y){
		this.X_y = X_y;
	 	n = X_y.size();
	 	p = X_y.get(0).length - 1;
	}
	
	public void setCovSplitPrior(double[] cov_split_prior){
		this.cov_split_prior = cov_split_prior;
	}
	
	public void setNumGibbsBurnIn(int num_gibbs_burn_in){
		this.num_gibbs_burn_in = num_gibbs_burn_in;
	}	

	public void setNumTrees(int num_trees){
		this.num_trees = num_trees;
	}
	
	public void setSampleVarY(double sample_var_y){
		this.sample_var_y = sample_var_y;
	}
	
	public void setAlpha(double alpha){
		this.alpha = alpha;
	}
	
	public void setBeta(double beta){
		this.beta = beta;
	}	
	
	public void setK(double hyper_k) {
		this.hyper_k = hyper_k;
	}

	public void setQ(double hyper_q) {
		this.hyper_q = hyper_q;
	}

	public void setNU(double hyper_nu) {
		this.hyper_nu = hyper_nu;
	}	
	
	
	public void setProbGrow(double prob_grow) {
		this.prob_grow = prob_grow;
	}

	public void setProbPrune(double prob_prune) {
		this.prob_prune = prob_prune;
	}	

	public void setProbChange(double prob_change) {
		//this does nothing
	}
	
	public void setVerbose(boolean verbose){
		this.verbose = verbose;
	}
	
	public void setNumCores(int num_cores){
		this.num_cores = num_cores;
	}
	
	public void setMemCacheForSpeed(boolean mem_cache_for_speed){
		this.mem_cache_for_speed = mem_cache_for_speed;
	}
	
	public boolean isDestroyed(){		
		return destroyed;
	}
	
	/** Must be implemented, but does nothing */
	public void StopBuilding() {}
	
}
