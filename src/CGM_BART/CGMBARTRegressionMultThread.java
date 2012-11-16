package CGM_BART;

import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


public class CGMBARTRegressionMultThread extends Classifier {
	private static final long serialVersionUID = -4537075714317768756L;
	
	private static final int DEFAULT_NUM_CORES = 1;//Runtime.getRuntime().availableProcessors() - 1;
	
	private int num_cores;
	
	private CGMBARTRegression[] bart_gibbs_chain_threads;
	protected CGMBARTTreeNode[][] gibbs_samples_of_cgm_trees_after_burn_in;
	protected double[] gibbs_samples_of_sigsq_after_burn_in;
	
	private int num_gibbs_burn_in;
	private int num_gibbs_total_iterations;
	private int total_iterations_multithreaded;

	private ArrayList<double[]> gibbs_samples_of_sigsq_hetero_aggregated;

	public CGMBARTRegressionMultThread(int num_cores){
		this.num_cores = num_cores;
		SetupBARTModels();
	}
	
	public CGMBARTRegressionMultThread(){
		num_cores = DEFAULT_NUM_CORES;
		SetupBARTModels();
	}
	
	private void SetupBARTModels() {
		bart_gibbs_chain_threads = new CGMBARTRegression[num_cores];
		for (int t = 0; t < num_cores; t++){
			CGMBARTRegression bart = new CGMBARTRegression();
			bart.setThreadNum(t);
			bart_gibbs_chain_threads[t] = bart;
		}	
		//we need to set defaults her
		num_gibbs_burn_in = CGMBART_01_base.DEFAULT_NUM_GIBBS_BURN_IN;
		num_gibbs_total_iterations = CGMBART_01_base.DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS;
		setNumGibbsTotalIterations(num_gibbs_total_iterations);		
	}
	


	@Override
	public void Build() {
		//run a build on all threads
		BuildOnAllThreads();
		//once it's done, now put together the chains
		ConstructBurnedChainForTreesAndSigsq();	
	}	
	
	protected void ConstructBurnedChainForTreesAndSigsq() {
		gibbs_samples_of_cgm_trees_after_burn_in = new CGMBARTTreeNode[numSamplesAfterBurning()][bart_gibbs_chain_threads[0].num_trees];
		gibbs_samples_of_sigsq_after_burn_in = new double[numSamplesAfterBurning()];

		System.out.print("burning and aggregating chains from all threads...");
		//go through each thread and get the tail and put them together
		for (int t = 0; t < num_cores; t++){
			CGMBARTRegression bart_model = bart_gibbs_chain_threads[t];
			for (int i = num_gibbs_burn_in; i < total_iterations_multithreaded; i++){
				gibbs_samples_of_cgm_trees_after_burn_in[i - num_gibbs_burn_in] = bart_model.gibbs_samples_of_cgm_trees[i];
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
			}
	    	);
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
	 	
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].setData(X_y);
		}	
	}
	
	public void setNumGibbsBurnIn(int num_gibbs_burn_in){
		this.num_gibbs_burn_in = num_gibbs_burn_in;
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].num_gibbs_burn_in = num_gibbs_burn_in;
		}
	}
	
	public void setNumGibbsTotalIterations(int num_gibbs_total_iterations){
		this.num_gibbs_total_iterations = num_gibbs_total_iterations;
		total_iterations_multithreaded = num_gibbs_burn_in + (int)Math.ceil((num_gibbs_total_iterations - num_gibbs_burn_in) / (double) num_cores);
		System.out.println("total_iterations_multithreaded: " + total_iterations_multithreaded);
		
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].num_gibbs_total_iterations = total_iterations_multithreaded;
		}
	}	

	public void setNumTrees(int m){
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].num_trees = m;
		}
	}
	
	public void setPrintOutEveryNIter(int print_out_every){
		CGMBART_01_base.PrintOutEvery = print_out_every;
	}
	
	
	public void setAlpha(double alpha){
		CGMBART_01_base.ALPHA = alpha;
	}
	
	public void setBeta(double beta){
		CGMBART_01_base.BETA = beta;
	}	
	
	public void setNumCores(Integer num_cores){
		if (num_cores != null){
			this.num_cores = num_cores;
		}
	}

	@Override
	protected void FlushData() {
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].FlushData();
		}
	}

	public double Evaluate(double[] record) { //posterior sample median (it's what Rob uses)		
		return EvaluateViaSampAvg(record);
	}	
	
	public double EvaluateViaSampMed(double[] record) { //posterior sample average		
		return StatToolbox.sample_median(getGibbsSamplesForPrediction(record));
	}
	
	public double EvaluateViaSampAvg(double[] record) { //posterior sample average		
		return StatToolbox.sample_average(getGibbsSamplesForPrediction(record));
	}
	
	public int numSamplesAfterBurning(){
		return num_gibbs_total_iterations - num_gibbs_burn_in;
	}		

	protected double[] getGibbsSamplesForPrediction(double[] record){
//		System.out.println("eval record: " + record + " numtrees:" + this.bayesian_trees.size());
		//the results for each of the gibbs samples
		double[] y_gibbs_samples = new double[numSamplesAfterBurning()];	
		for (int i = 0; i < numSamplesAfterBurning(); i++){
			CGMBARTTreeNode[] cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in[i];
			double yt_i = 0;
			for (CGMBARTTreeNode tree : cgm_trees){ //sum of trees right?
				yt_i += tree.Evaluate(record);
			}
			//just make sure we switch it back to really what y is for the user
			y_gibbs_samples[i] = bart_gibbs_chain_threads[0].un_transform_y(yt_i);
		}
		return y_gibbs_samples;
	}
	
	protected double[] getPostPredictiveIntervalForPrediction(double[] record, double coverage){
		//get all gibbs samples sorted
		double[] y_gibbs_samples_sorted = getGibbsSamplesForPrediction(record);
		Arrays.sort(y_gibbs_samples_sorted);
		
		//calculate index of the CI_a and CI_b
		int n_bottom = (int)Math.round((1 - coverage) / 2 * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
		int n_top = (int)Math.round(((1 - coverage) / 2 + coverage) * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
//		System.out.print("getPostPredictiveIntervalForPrediction record = " + IOTools.StringJoin(record, ",") + "  Ng=" + y_gibbs_samples_sorted.length + " n_a=" + n_bottom + " n_b=" + n_top + " guess = " + Evaluate(record));
		double[] conf_interval = {y_gibbs_samples_sorted[n_bottom], y_gibbs_samples_sorted[n_top]};
//		System.out.println("  [" + conf_interval[0] + ", " + conf_interval[1] + "]");
		return conf_interval;
	}
	
	protected double[] get95PctPostPredictiveIntervalForPrediction(double[] record){
		return getPostPredictiveIntervalForPrediction(record, 0.95);
	}
	
	public double[] getAvgCountsByAttribute(){
		double[] avg_counts = new double[p];
//		for (int t = 0; t < num_cores; t++){
//			double[] avg_counts_by_thread = bart_gibbs_chain_threads.get(t).getAvgCountsByAttribute();
//			for (int j = 0; j < p; j++){
//				avg_counts[j] += avg_counts_by_thread[j] / num_cores;
//			}			
//		}		
		return avg_counts;
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
	
	public int[][] getCountForAttributesForEntireChain(){
		int[][] var_count_matrix = new int[gibbs_samples_of_cgm_trees_after_burn_in.length][p];
		
		for (int g = 0; g < gibbs_samples_of_cgm_trees_after_burn_in.length; g++){
			var_count_matrix[g] = getCountForAttributeInGibbsSample(g);
		}
		return var_count_matrix;
	}	

	public int[] getCountForAttributeInGibbsSample(int g) {
		int[] counts = new int[p];
		for (int j = 0; j < p; j++){
			int tot_for_attr_j = 0;
			for (CGMBARTTreeNode root_node : gibbs_samples_of_cgm_trees_after_burn_in[g]){
				tot_for_attr_j += root_node.numTimesAttrUsed(j);
			}			
			counts[j] = tot_for_attr_j;
		}
		
		return counts;
	}

	@Override
	public void StopBuilding() {
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].StopBuilding();
		}
	}
	
	public void setCovSplitPrior(double[] cov_split_prior){
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].setCovSplitPrior(cov_split_prior);
		}		
	}
	
	public void useHeteroskedasticity(){
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads[t].useHeteroskedasticity();
		}		
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
		return bart_gibbs_chain_threads[0].un_transform_sigsq(bart_gibbs_chain_threads[0].gibbs_samples_of_sigsq_hetero[g]);
	}	
}
