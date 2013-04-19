package CGM_BART;

import gnu.trove.list.array.TDoubleArrayList;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import OpenSourceExtensions.UnorderedPair;


public class CGMBARTRegressionMultThread extends Classifier implements Serializable {
	private static final long serialVersionUID = -4537075714317768756L;
	
	private static final int DEFAULT_NUM_CORES = 4;//Runtime.getRuntime().availableProcessors() - 1;
		
	protected static final int NUM_TREES_DEFAULT = 200;
	protected static final int NUM_GIBBS_BURN_IN_DEFAULT = 250;
	protected static final int NUM_GIBBS_TOTAL_ITERATIONS_DEFAULT = 2000; //this must be larger than the number of burn in!!!

	protected static double HYPER_ALPHA_DEFAULT = 0.95;
	protected static double HYPER_BETA_DEFUALT = 2; //see p271 in CGM10	
	protected static double HYPER_K_DEFAULT = 2.0;	
	protected static double HYPER_Q_DEFAULT = 0.9;
	protected static double HYPER_NU_DEFAULT = 3.0;
	protected static double PROB_GROW_DEFAULT = 2.5 / 9.0;
	protected static double PROB_PRUNE_DEFAULT = 2.5 / 9.0;
	protected static double PROB_CHANGE_DEFAULT = 4 / 9.0;	
	
	protected int num_cores;
	protected int num_trees;
	
	protected transient CGMBARTRegression[] bart_gibbs_chain_threads;
	protected CGMBARTTreeNode[][] gibbs_samples_of_cgm_trees_after_burn_in;
	
	private Double sample_var_y;
	protected int num_gibbs_burn_in;
	protected int num_gibbs_total_iterations;
	protected int total_iterations_multithreaded;

	//set all hyperparameters here
	protected double[] cov_split_prior;
	protected Double alpha;
	protected Double beta;
	protected Double hyper_k;
	protected Double hyper_q;
	protected Double hyper_nu;
	protected Double prob_grow;
	protected Double prob_prune;
	protected Double prob_change;
	protected boolean verbose = true;
	protected boolean mem_cache_for_speed = false;

	protected transient boolean use_heteroskedasticity;

	

	
	public CGMBARTRegressionMultThread(){
//		System.out.print("new CGMBARTRegressionMultThread()");		
		//we need to set defaults here		
		num_cores = DEFAULT_NUM_CORES;
		num_trees = NUM_TREES_DEFAULT;
		num_gibbs_burn_in = NUM_GIBBS_BURN_IN_DEFAULT;
		num_gibbs_total_iterations = NUM_GIBBS_TOTAL_ITERATIONS_DEFAULT;
		alpha = HYPER_ALPHA_DEFAULT;
		beta = HYPER_BETA_DEFUALT;
		hyper_k = HYPER_K_DEFAULT;
		hyper_q = HYPER_Q_DEFAULT;
		hyper_nu = HYPER_NU_DEFAULT;
		prob_grow = PROB_GROW_DEFAULT;
		prob_prune = PROB_PRUNE_DEFAULT;
		prob_change = PROB_CHANGE_DEFAULT;		
		setNumGibbsTotalIterations(num_gibbs_total_iterations);
	}
	
	protected void SetupBARTModels() {
//		System.out.print("begin SetupBARTModels()");
		bart_gibbs_chain_threads = new CGMBARTRegression[num_cores];
		for (int t = 0; t < num_cores; t++){
			CGMBARTRegression bart = new CGMBARTRegression();
			SetupBartModel(bart, t);
		}	
//		System.out.print("end SetupBARTModels()");
	}

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
		bart.setProbChange(prob_change);		
		//set thread num and data
		bart.setThreadNum(t);
		bart.setTotalNumThreads(num_cores);
		bart.setMemCacheForSpeed(mem_cache_for_speed);
		
		//set features
		if (cov_split_prior != null){
			bart.setCovSplitPrior(cov_split_prior);
		}
		if (use_heteroskedasticity){
			bart.useHeteroskedasticity();
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

	@Override
	public void Build() {
//		System.out.println("Build()");
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

	public void setData(ArrayList<double[]> X_y){
		this.X_y = X_y;
	 	n = X_y.size();
	 	p = X_y.get(0).length - 1;
	}
	
	public void setNumGibbsBurnIn(int num_gibbs_burn_in){
		this.num_gibbs_burn_in = num_gibbs_burn_in;
	}
	
	public void setNumGibbsTotalIterations(int num_gibbs_total_iterations){
		this.num_gibbs_total_iterations = num_gibbs_total_iterations;
		total_iterations_multithreaded = num_gibbs_burn_in + (int)Math.ceil((num_gibbs_total_iterations - num_gibbs_burn_in) / (double) num_cores);
//		System.out.println("num_gibbs_total_iterations: " + num_gibbs_total_iterations + " total_iterations_multithreaded: " + total_iterations_multithreaded + " num_cores: " + num_cores);
	}	

	public void setNumTrees(int num_trees){
//		System.out.print("setNumTrees()");
		this.num_trees = num_trees;
	}
	
	public void setSampleVarY(double sample_var_y){
		this.sample_var_y = sample_var_y;
	}
	
	public void setPrintOutEveryNIter(int print_out_every){
		CGMBART_01_base.PrintOutEvery = print_out_every;
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
		this.prob_change = prob_change;
	}
	
	public void setVerbose(boolean verbose){
		this.verbose = verbose;
	}
	
	public void setMemCacheForSpeed(boolean mem_cache_for_speed){
		this.mem_cache_for_speed = mem_cache_for_speed;
	}
	
	public void setNumCores(int num_cores){
//		System.out.print("setNumCores()");
		this.num_cores = num_cores;
	}

	@Override
	protected void FlushData() {
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].FlushData();
		}
	}

	public double Evaluate(double[] record) { //posterior sample median (it's what Rob uses)		
		return EvaluateViaSampAvg(record, 1);
	}	
	
	public double Evaluate(double[] record, int num_cores_evaluate) { //posterior sample median (it's what Rob uses)		
		return EvaluateViaSampAvg(record, num_cores_evaluate);
	}		
	
	public double EvaluateViaSampMed(double[] record, int num_cores_evaluate) { //posterior sample average		
		double[][] data = new double[1][record.length];
		data[0] = record;
		double[][] gibbs_samples = getGibbsSamplesForPrediction(data, num_cores_evaluate);
		return StatToolbox.sample_median(gibbs_samples[0]);
	}
	
	public double EvaluateViaSampAvg(double[] record, int num_cores_evaluate) { //posterior sample average		
		double[][] data = new double[1][record.length];
		data[0] = record;
		double[][] gibbs_samples = getGibbsSamplesForPrediction(data, num_cores_evaluate);
		return StatToolbox.sample_average(gibbs_samples[0]);
	}
	
	public int numSamplesAfterBurning(){
		return num_gibbs_total_iterations - num_gibbs_burn_in;
	}	
	
	/**
	 * Code is ugly and not decomped because it is optimized
	 * 
	 * 
	 * @param data
	 * @param num_cores
	 * @return
	 */
	protected double[][] getGibbsSamplesForPrediction(final double[][] data, final int num_cores){
		final int num_samples_after_burn_in = numSamplesAfterBurning();
		final CGMBARTRegression first_bart = bart_gibbs_chain_threads[0];
		
		final int n = data.length;
		final double[][] y_hat = new double[n][data[0].length];
//		System.out.println("getGibbsSamplesForPrediction n: " + n);
		
		if (num_cores == 1){
			for (int i = 0; i < n; i++){
				double[] y_gibbs_samples = new double[num_samples_after_burn_in];
				for (int g = 0; g < num_samples_after_burn_in; g++){
					CGMBARTTreeNode[] cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in[g];
					double yt_i = 0;
					for (int m = 0; m < num_trees; m++){ //sum of trees right?
						yt_i += cgm_trees[m].Evaluate(data[i]);
					}
					//just make sure we switch it back to really what y is for the user
					y_gibbs_samples[g] = first_bart.un_transform_y(yt_i);
//					System.out.println("y_gibbs_samples[g]: " + y_gibbs_samples[g]);
				}
				y_hat[i] = y_gibbs_samples;
			}			
		}
		else { //probably should put back executor service here for cleanliness
			Thread[] fixed_thread_pool = new Thread[num_cores];
			for (int t = 0; t < num_cores; t++){
				final int final_t = t;
				Thread thread = new Thread(){
					public void run(){
						for (int i = 0; i < n; i++){
							if (i % num_cores == final_t){
								double[] y_gibbs_samples = new double[num_samples_after_burn_in];
								for (int g = 0; g < num_samples_after_burn_in; g++){									
									CGMBARTTreeNode[] cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in[g];
									double yt_i = 0;
									for (int m = 0; m < num_trees; m++){ //sum of trees right?
										yt_i += cgm_trees[m].Evaluate(data[i]);
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
			for (int t = 0; t < num_cores; t++){
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
//		System.out.print("getPostPredictiveIntervalForPrediction record = " + IOTools.StringJoin(record, ",") + "  Ng=" + y_gibbs_samples_sorted.length + " n_a=" + n_bottom + " n_b=" + n_top + " guess = " + Evaluate(record));
		double[] conf_interval = {y_gibbs_samples_sorted[n_bottom], y_gibbs_samples_sorted[n_top]};
//		System.out.println("  [" + conf_interval[0] + ", " + conf_interval[1] + "]");
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
		//what's the SIGSQ?
//		for (int g = 0; g < sigsqs_to_export.size(); g++){
//			System.out.println("g = " + g + " sigsq: " + sigsqs_to_export.get(g));			
//		}
//		System.out.println("MEAN: " + StatToolbox.sample_average(sigsqs_to_export.toArray()));
		
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
	
//	public int[][] getCountForAttributesForEntireChain(){
//		int[][] var_count_matrix = new int[gibbs_samples_of_cgm_trees_after_burn_in.length][p];
//		
//		for (int g = 0; g < gibbs_samples_of_cgm_trees_after_burn_in.length; g++){
//			var_count_matrix[g] = getCountForAttributeInGibbsSample(g);
//		}
//		return var_count_matrix;
//	}	
	
	public double[] getAttributeProps(final int num_cores, final String type) {
		int[][] variable_counts_all_gibbs = getCountsForAllAttribute(num_cores, type);
		double[] attribute_counts = new double[p];
		for (int g = 0; g < num_gibbs_total_iterations - num_gibbs_burn_in; g++){
			attribute_counts = Tools.add_arrays(attribute_counts, variable_counts_all_gibbs[g]);
		}
		
		return Tools.scale_array(attribute_counts); //will turn it into proportions
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
		
//		ExecutorService get_count_for_attribute_pool = Executors.newFixedThreadPool(num_cores);
//		
//		for (int c = 0; c < num_cores; c++){
//			final int cf = c;
//			get_count_for_attribute_pool.execute(new Runnable(){
//				public void run() {
//					for (int g = 0; g < num_gibbs_total_iterations - num_gibbs_burn_in; g++){
//						if (g % num_cores == cf){
//							final CGMBARTTreeNode[] trees = gibbs_samples_of_cgm_trees_after_burn_in[g];			
//							
//							int[] total_for_trees = new int[p]; //each entry in this array is the number of times attr j was used for all m trees in this gibbs sample
//							for (CGMBARTTreeNode tree : trees){	
//								if (type.equals("splits")){
//									tree.numTimesAttrUsed(total_for_trees);
//								}
//								else if (type.equals("trees")){
//									int[] total_for_trees_temp = new int[p];
//									tree.attrUsed(total_for_trees_temp);
//									total_for_trees = Tools.add_arrays(total_for_trees_temp, total_for_trees);
//								}
//							}
//							counts[g] = total_for_trees;
//						}
//					}					
//				}
//			});
//		}
//		
//		//now join em up and ship out the result
//		get_count_for_attribute_pool.shutdown();
//		try {	         
//			get_count_for_attribute_pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS); //effectively infinity
//	    } catch (InterruptedException ignored){}	
		
		return variable_counts_all_gibbs;
	}

	@Override
	public void StopBuilding() {
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].StopBuilding();
		}
	}
	
	public void setCovSplitPrior(double[] cov_split_prior){
		this.cov_split_prior = cov_split_prior;
	}
	
	public void useHeteroskedasticity(){
		use_heteroskedasticity = true;
		System.out.println("using heteroskedastic BART");		
	}

	public double[] getSigsqsByGibbsSample(int g){
//		if (gibbs_samples_of_sigsq_hetero_aggregated == null){
//			gibbs_samples_of_sigsq_hetero_aggregated = new ArrayList<double[]>(numSamplesAfterBurning());
//			for (int t = 0; t < num_cores; t++){
//				for (int i = 0; i < total_iterations_multithreaded - num_gibbs_burn_in; i++){
//					double[] sigsqs_gibbs = bart_gibbs_chain_threads[t].gibbs_samples_of_sigsq_hetero[i];
//					System.out.println("sigsqs_gibbs: " + Tools.StringJoin(sigsqs_gibbs));
//					gibbs_samples_of_sigsq_hetero_aggregated.add(sigsqs_gibbs);
//				}
//			}			
//			
//		}
//		System.out.println("getSigsqsByGibbsSample  bart_gibbs_chain_threads[0]: " + bart_gibbs_chain_threads[0] + " g = " + g);
		return bart_gibbs_chain_threads[0].un_transform_sigsq(bart_gibbs_chain_threads[0].gibbs_samples_of_sigsq_hetero[g]);
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
	}
	

	public int[][] getInteractionCounts(int num_cores){
		final int[][] interaction_count_matrix = new int[p][p];
//		ExecutorService interaction_count_getter_pool = Executors.newFixedThreadPool(num_cores);
		
		for (int g = 0; g < gibbs_samples_of_cgm_trees_after_burn_in.length; g++){
			final int gf = g;
//	    	interaction_count_getter_pool.execute(new Runnable(){
//				public void run() {
					CGMBARTTreeNode[] trees = gibbs_samples_of_cgm_trees_after_burn_in[gf];
					
					for (CGMBARTTreeNode tree : trees){
						//get the set of pairs of interactions
						HashSet<UnorderedPair<Integer>> set_of_interaction_pairs = new HashSet<UnorderedPair<Integer>>(p * p);
						//find all interactions
						tree.findInteractions(set_of_interaction_pairs);
//						Tools.print_unordered_pair_set(set_of_interaction_pairs);
						//now tabulate these interactions in our count matrix
//						synchronized(interaction_count_matrix){
							for (UnorderedPair<Integer> pair : set_of_interaction_pairs){
//								System.out.println("interaction: " + pair.getFirst() + " " + pair.getSecond());
								interaction_count_matrix[pair.getFirst()][pair.getSecond()]++; 
							}
//						}
					}
//				}
//			});			
//			
		}
//		interaction_count_getter_pool.shutdown();
//		try {	         
//	         interaction_count_getter_pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS); //effectively infinity
//		} catch (InterruptedException ignored){}
		
		return interaction_count_matrix;
	}

//	int[][] ics = getInteractionCounts(num_cores);
//	System.out.println("interaction counts");
//	for (int j1 = 0; j1 < p; j1++){
//		System.out.println(Tools.StringJoin(ics[j1], "\t"));
//	}
}
