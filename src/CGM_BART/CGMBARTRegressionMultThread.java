package CGM_BART;

import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


public class CGMBARTRegressionMultThread extends Classifier {
	private static final long serialVersionUID = -4537075714317768756L;
	
	private static final int DEFAULT_NUM_CORES = 3;//Runtime.getRuntime().availableProcessors() - 1;
	
	private int num_cores;
	private int num_trees;
	
	private CGMBARTRegression[] bart_gibbs_chain_threads;
	protected CGMBARTTreeNode[][] gibbs_samples_of_cgm_trees_after_burn_in;
	protected double[] gibbs_samples_of_sigsq_after_burn_in;
	
	private double sample_var_y;
	private int num_gibbs_burn_in;
	private int num_gibbs_total_iterations;
	private int total_iterations_multithreaded;

	private double[] cov_split_prior;

	private boolean use_heteroskedasticity;

	
	
	public CGMBARTRegressionMultThread(){
//		System.out.print("new CGMBARTRegressionMultThread()");		
		//we need to set defaults here		
		num_cores = DEFAULT_NUM_CORES;
		num_trees = CGMBART_01_base.DEFAULT_NUM_TREES;
		num_gibbs_burn_in = CGMBART_01_base.DEFAULT_NUM_GIBBS_BURN_IN;
		num_gibbs_total_iterations = CGMBART_01_base.DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS;
		setNumGibbsTotalIterations(num_gibbs_total_iterations);			
	}
	
	private void SetupBARTModels() {
//		System.out.print("begin SetupBARTModels()");
		bart_gibbs_chain_threads = new CGMBARTRegression[num_cores];
		for (int t = 0; t < num_cores; t++){
			CGMBARTRegression bart = new CGMBARTRegression();
			//now set specs on each of the bart models
			bart.num_trees = num_trees;
			bart.num_gibbs_total_iterations = total_iterations_multithreaded;
			bart.num_gibbs_burn_in = num_gibbs_burn_in;
			bart.sample_var_y = sample_var_y;
			bart.setThreadNum(t);
			bart.setData(X_y);
			//set features
			if (cov_split_prior != null){
				bart.setCovSplitPrior(cov_split_prior);
			}
			if (use_heteroskedasticity){
				bart.useHeteroskedasticity();
			}
			bart_gibbs_chain_threads[t] = bart;
		}	
//		System.out.print("end SetupBARTModels()");
	}

	@Override
	public void Build() {
//		System.out.println("Build()");
		SetupBARTModels();
		//run a build on all threads
		BuildOnAllThreads();
		//once it's done, now put together the chains
		ConstructBurnedChainForTreesAndOtherInformation();	
		
//		int[][] depths = getDepthsForTreesInGibbsSampAfterBurnIn(0);
//		for (int g = 0 ; g < depths.length; g++){
//			System.out.println("depths for gibbs sample " + g + ": " + Tools.StringJoin(depths[g]));
//		}
	}	
	
	protected void ConstructBurnedChainForTreesAndOtherInformation() {
		gibbs_samples_of_cgm_trees_after_burn_in = new CGMBARTTreeNode[numSamplesAfterBurning()][num_trees];
		gibbs_samples_of_sigsq_after_burn_in = new double[numSamplesAfterBurning()];

		System.out.print("burning and aggregating chains from all threads... ");
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
				gibbs_samples_of_sigsq_after_burn_in[i - num_gibbs_burn_in] = bart_model.gibbs_samples_of_sigsq[i];
			}			
		}				
		System.out.print("done\n");
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
//		System.out.print("setAlpha()");
		CGMBART_01_base.ALPHA = alpha;
	}
	
	public void setBeta(double beta){
//		System.out.print("setBeta()");
		CGMBART_01_base.BETA = beta;
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
				}
				y_hat[i] = y_gibbs_samples;
			}			
		}
		else {
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

	public int[] getCountForAttributeInGibbsSample(int g, int num_cores_count) {
		final int[] counts = new int[p];		
		final CGMBARTTreeNode[] cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in[g];
		
		ExecutorService get_count_for_attribute_pool = Executors.newFixedThreadPool(num_cores_count);
		
		for (int j = 0; j < p; j++){
			final int final_j = j;
			get_count_for_attribute_pool.execute(new Runnable(){
				public void run() {
					int tot_for_attr_j = 0;
					for (CGMBARTTreeNode root_node : cgm_trees){
						tot_for_attr_j += root_node.numTimesAttrUsed(final_j);
					}		
					counts[final_j] = tot_for_attr_j;					
				}
			});

		}
		
		//now join em up and ship out the result
		get_count_for_attribute_pool.shutdown();
		try {	         
			get_count_for_attribute_pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS); //effectively infinity
	    } catch (InterruptedException ignored){}	
		
		return counts;
	}

	@Override
	public void StopBuilding() {
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].StopBuilding();
		}
	}
	
	public void setCovSplitPrior(double[] cov_split_prior){
		this.cov_split_prior = cov_split_prior;
		System.out.println("using BART with covariate importance prior");		
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
		return bart_gibbs_chain_threads[thread_num - 1].getDepthsForTrees(num_gibbs_burn_in + 1, total_iterations_multithreaded);
	}	
	
	public int[][] getNumNodesAndLeavesForTreesInGibbsSampAfterBurnIn(int thread_num){
		return bart_gibbs_chain_threads[thread_num - 1].getNumNodesAndLeavesForTrees(num_gibbs_burn_in + 1, total_iterations_multithreaded);
	}	
	
	public void destroy(){
		bart_gibbs_chain_threads = null;
		gibbs_samples_of_cgm_trees_after_burn_in = null;
		gibbs_samples_of_sigsq_after_burn_in = null;
		cov_split_prior = null;
	}

}
