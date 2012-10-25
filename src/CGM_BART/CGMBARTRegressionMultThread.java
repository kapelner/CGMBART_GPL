package CGM_BART;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


public class CGMBARTRegressionMultThread extends Classifier {
	private static final long serialVersionUID = -4537075714317768756L;
	
	private static final int DEFAULT_NUM_CORES = 2;//Runtime.getRuntime().availableProcessors() - 1;
	private int num_cores;
	
	private ArrayList<CGMBARTRegression> bart_gibbs_chain_threads;
	protected ArrayList<ArrayList<CGMBARTTreeNode>> gibbs_samples_of_cgm_trees_after_burn_in;
	protected ArrayList<Double> gibbs_samples_of_sigsq_after_burn_in;
	
	private int num_gibbs_burn_in;
	private int num_gibbs_total_iterations;
	private int total_iterations_multithreaded;

	public CGMBARTRegressionMultThread(int num_cores){
		this.num_cores = num_cores;
		SetupBARTModels();
	}
	
	public CGMBARTRegressionMultThread(){
		num_cores = DEFAULT_NUM_CORES;
		SetupBARTModels();
	}
	
	private void SetupBARTModels() {
		bart_gibbs_chain_threads = new ArrayList<CGMBARTRegression>(num_cores);
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads.add(new CGMBARTRegression());
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
	
	private void ConstructBurnedChainForTreesAndSigsq() {
		gibbs_samples_of_cgm_trees_after_burn_in = new ArrayList<ArrayList<CGMBARTTreeNode>>(numSamplesAfterBurning());
		gibbs_samples_of_sigsq_after_burn_in = new ArrayList<Double>(numSamplesAfterBurning());

		//go through each thread and get the tail and put them together
		for (int t = 0; t < num_cores; t++){
			CGMBARTRegression bart_model = bart_gibbs_chain_threads.get(t);
			for (int i = num_gibbs_burn_in; i < total_iterations_multithreaded; i++){
				gibbs_samples_of_cgm_trees_after_burn_in.add(bart_model.gibbs_samples_of_cgm_trees.get(i));
				gibbs_samples_of_sigsq_after_burn_in.add(bart_model.gibbs_samples_of_sigsq.get(i));
			}			
		}		
		
	}

	private void BuildOnAllThreads(){
		ExecutorService bart_gibbs_chain_pool = Executors.newFixedThreadPool(num_cores);
		for (int t = 0; t < num_cores; t++){
			final int tf = t;
	    	bart_gibbs_chain_pool.execute(new Runnable(){
				public void run() {
					bart_gibbs_chain_threads.get(tf).Build();
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
		super.setData(X_y);
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads.get(t).setData(X_y);
		}	
	}
	
	public void setNumGibbsBurnIn(int num_gibbs_burn_in){
		this.num_gibbs_burn_in = num_gibbs_burn_in;
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads.get(t).num_gibbs_burn_in = num_gibbs_burn_in;
		}
	}
	
	public void setNumGibbsTotalIterations(int num_gibbs_total_iterations){
		this.num_gibbs_total_iterations = num_gibbs_total_iterations;
		total_iterations_multithreaded = num_gibbs_burn_in + (int)Math.round((num_gibbs_total_iterations - num_gibbs_burn_in) / (double) num_cores);
		System.out.println("total_iterations_multithreaded: " + total_iterations_multithreaded);
		
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads.get(t).num_gibbs_total_iterations = total_iterations_multithreaded;
		}
	}	

	public void setNumTrees(int m){
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads.get(t).num_trees = m;
		}
	}
	
	public void setPrintOutEveryNIter(int print_out_every){
		CGMBART_01_base.PrintOutEvery = print_out_every;
	}
	
	
	public void setAlpha(double alpha){
		CGMBART_01_base.ALPHA = alpha;
	}
	
	public void setNumCores(int num_cores){
		this.num_cores = num_cores;
	}

	@Override
	protected void FlushData() {
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads.get(t).FlushData();
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
			ArrayList<CGMBARTTreeNode> cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in.get(i);
			double yt_i = 0;
			for (CGMBARTTreeNode tree : cgm_trees){ //sum of trees right?
				yt_i += tree.Evaluate(record);
			}
			//just make sure we switch it back to really what y is for the user
			y_gibbs_samples[i] = bart_gibbs_chain_threads.get(0).un_transform_y(yt_i);
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

	@Override
	public void StopBuilding() {
		for (int t = 0; t < num_cores; t++){
			bart_gibbs_chain_threads.get(t).StopBuilding();
		}
	}

}
