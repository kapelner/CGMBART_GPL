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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import CGM_Statistics.*;
import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentStatistics.Classifier;
import GemIdentTools.IOTools;
import GemIdentView.JProgressBarAndLabel;

@SuppressWarnings("serial")
public abstract class CGMBART extends Classifier implements Serializable  {

	//do not set this to FALSE!!! The whole thing will break...
	protected static final boolean TRANSFORM_Y = true;
	protected static final int DEFAULT_NUM_TREES = 1;
	//this burn in number needs to be computed via some sort of moving average or time series calculation
	protected static final int DEFAULT_NUM_GIBBS_BURN_IN = 200;
	protected static final int DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS = 400; //this must be larger than the number of burn in!!!
	
//	protected static final double GIBBS_THIN_RATIO = 0.1;
	
	protected static PrintWriter y_and_y_trans;
	protected static PrintWriter sigsqs;
	protected static PrintWriter other_debug;
	protected static PrintWriter sigsqs_draws;
	protected static PrintWriter tree_liks;
	protected static PrintWriter remainings;
	protected static PrintWriter evaluations;
	protected static boolean TREE_ILLUST = false;
	protected static final boolean WRITE_DETAILED_DEBUG_FILES = false;
	
//	static {
//		Classifier.writeToDebugLog();
//	}

	static {
		try {			
			output = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "output" + DEBUG_EXT)));
			TreeIllustration.DeletePreviousTreeIllustrations();
			other_debug = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "other_debug" + DEBUG_EXT)));
			y_and_y_trans = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "y_and_y_trans" + DEBUG_EXT)));
			sigsqs = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "sigsqs" + DEBUG_EXT)));
			sigsqs.println("sample_num,sigsq");
			sigsqs_draws = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "sigsqs_draws" + DEBUG_EXT)));
			double[] simu = new double[1000];
			for (int i = 1; i <= 1000; i++){
				simu[i-1] = i;
			}			
			sigsqs_draws.println("sample_num,nu,lambda,n,sse,realization,corr," + IOTools.StringJoin(simu, ","));			
			tree_liks = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "tree_liks" + DEBUG_EXT)));
			evaluations = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "evaluations" + DEBUG_EXT)));
			remainings = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "remainings" + DEBUG_EXT)));
			tree_liks.print("sample_num,");
			for (int t = 0; t < DEFAULT_NUM_TREES; t++){
				tree_liks.print("t_" + t + "_lik,t_" + t + "_id,");
			}
			tree_liks.print("\n");
			TreeIllustration.DeletePreviousTreeIllustrations();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/** the actual list of trees */
	protected ArrayList<ArrayList<CGMTreeNode>> gibbs_samples_of_cgm_trees;
	protected ArrayList<ArrayList<CGMTreeNode>> gibbs_samples_of_cgm_trees_after_burn_in;
	/** the variance of the errors */
	protected ArrayList<Double> gibbs_samples_of_sigsq;
	protected ArrayList<Double> gibbs_samples_of_sigsq_after_burn_in;
	/** information about the response variable */
	protected double y_min;
	protected double y_max;
	protected double y_range_sq;
	/** the current # of trees */
	protected int m;
	protected int num_gibbs_burn_in;
	protected int num_gibbs_total_iterations;	
	/** all the hyperparameters */
	protected double hyper_mu_mu;
	protected double hyper_sigsq_mu;
	protected double hyper_nu;
	protected double hyper_lambda;
	/** we will use the tree prior builder to initally build the trees and later in the gibbs sampler as well */
	protected CGMTreePriorBuilder tree_prior_builder;
	/** stuff during the build run time that we can access and look at */
	protected int gibb_sample_i;
	protected double[][] all_tree_liks;
	/** if the user pressed stop, we can cancel the Gibbs Sampling to unlock the CPU */
	protected boolean stop_bit;	
	protected static Integer PrintOutEvery = null;
	/** during debugging, we may want to fix sigsq */
	protected double fixed_sigsq;
	
	/** Serializable happy */
	public CGMBART(){}
	
	/**
	 * Constructs the BART classifier for regression. We rely on the SetupClassification class to set the raw data
	 * 
	 * @param datumSetup
	 * @param buildProgress
	 */
	public CGMBART(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
//		System.out.println("CGMBART constructor");
		m = DEFAULT_NUM_TREES;
		num_gibbs_burn_in = DEFAULT_NUM_GIBBS_BURN_IN;
		num_gibbs_total_iterations = DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS;
	}	
	
	public void setNumTrees(int m){
		this.m = m;
	}
	
	public void setSigsq(double fixed_sigsq){
		this.fixed_sigsq = fixed_sigsq;
	}
	
	public void printTreeIllustations(){
		TREE_ILLUST = true;
	}
	
	public void setPrintOutEveryNIter(int print_out_every){
		PrintOutEvery = print_out_every;
	}
	
	public void setNumGibbsBurnIn(int num_gibbs_burn_in){
		this.num_gibbs_burn_in = num_gibbs_burn_in;
	}
	
	public void setNumGibbsTotalIterations(int num_gibbs_total_iterations){
		this.num_gibbs_total_iterations = num_gibbs_total_iterations;
	}	
	
	public void setAlpha(double alpha){
		CGMBARTPriorBuilder.ALPHA = alpha;
	}
	
	public void setBeta(double beta){
		CGMBARTPriorBuilder.BETA = beta;
	}	

	@Override
	public void Build() {
		//first establish the hyperparams sigsq_mu, nu, lambda
		calculateHyperparameters();
		all_tree_liks = new double[m][num_gibbs_total_iterations + 1];

		//now generate a prior builder... used in any implementation
		tree_prior_builder = new CGMBARTPriorBuilder(X_y, p); //this is technically incorrect since we should create the initial trees by dividing y by m... but who cares, it won't make a difference anyway

		//now initialize the gibbs sampler array for trees and error variances
		gibbs_samples_of_cgm_trees = new ArrayList<ArrayList<CGMTreeNode>>(num_gibbs_total_iterations);
		gibbs_samples_of_cgm_trees_after_burn_in = new ArrayList<ArrayList<CGMTreeNode>>(num_gibbs_total_iterations - num_gibbs_burn_in);
		gibbs_samples_of_cgm_trees.add(null);
		gibbs_samples_of_sigsq = new ArrayList<Double>(num_gibbs_total_iterations);	
		gibbs_samples_of_sigsq_after_burn_in = new ArrayList<Double>(num_gibbs_total_iterations - num_gibbs_burn_in);	

		//this section is different for the different BART implementations
		//but it basically does all the Gibbs sampling
		CoreBuild();
		

//		for (int n_g = 0; n_g < num_gibbs_total_iterations; n_g++){
//			System.out.println("n_g: " + n_g + " name " + gibbs_samples_of_cgm_trees.get(n_g));
//		}
		for (int b = 1; b <= 7; b++){
			System.out.println("all vals of mu1: " + getMuValuesForAllItersByTreeAndLeaf(0, b));
		}

		//make sure debug files are closed
		CloseDebugFiles();

//		System.err.println("num gibbs samples of cgm trees: " + gibbs_samples_of_cgm_trees.size());
//		for (int i = 1; i <= gibbs_samples_of_cgm_trees.size(); i++){
//			ArrayList<CGMTreeNode> trees = gibbs_samples_of_cgm_trees.get(i - 1);
//			if (trees == null){
//				System.out.println("gibbs sample " + i + " null trees");
//			}
//			else {
//				for (int t = 1; t <= trees.size(); t++){
//					CGMTreeNode tree = trees.get(t - 1);
//					System.out.println("gibbs sample " + i + " tree " + t + " depth=" + tree.deepestNode() + " b=" + tree.numLeaves());
//				}
//			}
//		}		
//		System.err.println("err");
		
		//now we burn and thin the chains for each param
		BurnTreeAndSigsqChain();
		//for some reason we don't thin... I don't really understand why... has to do with autocorrelation and stickiness?
//		ThinBothChains();
		
		//make sure you can do stuff from R
//		getGibbsSamplesSigsqs();
//		for (int i = 0; i < num_gibbs_total_iterations; i++){
//			getNumNodesForTreesInGibbsSamp(i);
//			getRootSplits(i);
//			getDepthsForTreesInGibbsSamp(i);
//		}
//		for (int t = 0; t < m; t++){
//			getLikForTree(t);
//		}
//		
//		getGibbsSamplesForPrediction(X_y.get(0));

	}
	
	public int currentGibbsSampleIteration(){
		return gibb_sample_i;
	}
	
	public boolean gibbsFinished(){
		return gibb_sample_i >= num_gibbs_total_iterations;
	}

	protected abstract void CoreBuild();

	private void BurnTreeAndSigsqChain() {		
		for (int i = num_gibbs_burn_in; i < num_gibbs_total_iterations; i++){
			gibbs_samples_of_cgm_trees_after_burn_in.add(gibbs_samples_of_cgm_trees.get(i));
			gibbs_samples_of_sigsq_after_burn_in.add(gibbs_samples_of_sigsq.get(i));
		}	
		System.out.println("BurnTreeAndSigsqChain gibbs_samples_of_sigsq_after_burn_in length = " + gibbs_samples_of_sigsq_after_burn_in.size());
	}	
	
	@Override
	public double Evaluate(double[] record) { //posterior sample median (it's what Rob uses)		
		return EvaluateViaSampMed(record);
	}	
	
	public double EvaluateViaSampMed(double[] record) { //posterior sample average		
		return StatToolbox.sample_median(getGibbsSamplesForPrediction(record));
	}
	
	public double EvaluateViaSampAvg(double[] record) { //posterior sample average		
		return StatToolbox.sample_average(getGibbsSamplesForPrediction(record));
	}		
	
	public int numSamplesAfterBurningAndThinning(){
		return num_gibbs_total_iterations - num_gibbs_burn_in;
	}

	protected double[] getGibbsSamplesForPrediction(double[] record){
//		System.out.println("eval record: " + record + " numtrees:" + this.bayesian_trees.size());
		//the results for each of the gibbs samples
		double[] y_gibbs_samples = new double[numSamplesAfterBurningAndThinning()];	
		for (int i = 0; i < numSamplesAfterBurningAndThinning(); i++){
			ArrayList<CGMTreeNode> cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in.get(i);
			double yt_i = 0;
			for (CGMTreeNode tree : cgm_trees){ //sum of trees right?
				yt_i += tree.Evaluate(record);
			}
			//just make sure we switch it back to really what y is for the user
			y_gibbs_samples[i] = un_transform_y(yt_i);
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

	protected void runGibbsSamplerForTreesAndSigsqOnce(final int sample_num) {		
//		System.out.println("\n\nRunning BART Gibbs sampler, iteration " + sample_num + "\n\n");
		tree_liks.print(sample_num + ",");
		
		//now, we cycle over each tree and update it according to formulas 15, 16 on p274
		//we update it via a two stage process:
		//we first update the tree by running one iteration of CGM98 which involves a grow, prune, change, or swap
		//then we set the prediction values (ie the Mij's) via a draw from a normal distribution
		//this is essential, because we need to calculate residuals for the next iteration
		final ArrayList<CGMTreeNode> cgm_trees = new ArrayList<CGMTreeNode>(m);
		
		final TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(sample_num);	
		
		for (int t = 0; t < m; t++){
			//first sample T_j | R_j, \sigma
//			System.out.println("Sampling T_" + (t + 1) + " / " + m + " iter " + sample_num);
			CGMTreeNode tree = SampleTreeByCalculatingRemainingsAndDrawingFromTreeDist(sample_num - 1, t, tree_array_illustration);
			
			//we gotta use the previous crop of trees
			cgm_trees.add(tree);
			//now set the new trees in the gibbs sample pantheon, keep updating it...
			gibbs_samples_of_cgm_trees.set(sample_num, cgm_trees);
			
			tree_array_illustration.AddTree(tree);
			
			//now sample from M_j | T_j, R_j, \sigma by just assigning leaves a draw from the leaf posterior
			if (t + 1 == 1){
				System.out.println("Sampling M_" + (t + 1) + "/" + m + " iter " + sample_num + "/" + num_gibbs_total_iterations);
			}
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(tree, gibbs_samples_of_sigsq.get(sample_num - 1));
			
			if (stop_bit){
				return;
			}
		}		
				
		//DEBUG STUFF
		if (WRITE_DETAILED_DEBUG_FILES){	
			remainings.println((sample_num) + ",,y," + IOTools.StringJoin(y_trans, ","));
			
			ArrayList<CGMTreeNode> current_trees = gibbs_samples_of_cgm_trees.get(sample_num);
			for (int t = 0; t < m; t++){
				CGMTreeNode tree = current_trees.get(t);
				ArrayList<String> all_results = new ArrayList<String>(n);
				for (int i = 0; i < n; i++){
					all_results.add("" + tree.Evaluate(X_y.get(i)));
				} 
				evaluations.println(sample_num + "," + t + "," + tree.stringID() + "," + IOTools.StringJoin(all_results, ","));
			}	
			evaluations.println((sample_num) + ",,y," + IOTools.StringJoin(y_trans, ","));
		}
//		final Thread illustrator_thread = new Thread(){
//			public void run(){
//		if (Math.random() < 0.0333){
			if (TREE_ILLUST){
				tree_array_illustration.CreateIllustrationAndSaveImage();
			}
//		}
//			}
//		};
//		illustrator_thread.start();
		
		tree_liks.print("\n");			
		
		//now set the new trees in the gibbs sample pantheon
		gibbs_samples_of_cgm_trees.add(sample_num, cgm_trees);
		
		//okay now we have to sample a posterior sigma given all the trees and add that to the gibbs pantheon
		double current_sigsq = samplePosteriorSigsq(sample_num);
//		System.out.println("current_sigsq = " + current_sigsq);
		
		
		gibbs_samples_of_sigsq.add(sample_num, current_sigsq);
		if (TRANSFORM_Y){
			sigsqs.println(sample_num + "," + gibbs_samples_of_sigsq.get(sample_num) * y_range_sq);	
		}
		else {
			sigsqs.println(sample_num + "," + gibbs_samples_of_sigsq.get(sample_num));
		}
		
		//note: after we've made the predictions and the image, we can flush the data
//		new Thread(){
//			public void run(){
//				try {
//					illustrator_thread.join();
//				} catch (InterruptedException e) {
//					e.printStackTrace();
//				}
//				for (CGMTreeNode tree : cgm_trees){
//					tree.flushNodeData();	
//				}
//			}
//		};
		
		for (CGMTreeNode tree : cgm_trees){
			tree.flushNodeData();	
		}		
		
		
		//now close and open all debug
		if (Math.random() < 0.0333){
			CloseDebugFiles();
			OpenDebugFiles();
		}
	
	}
	
	public double[] getGibbsSamplesSigsqs(){
		double[] sigsqs_to_export = new double[gibbs_samples_of_sigsq.size()];
		for (int n_g = 0; n_g < gibbs_samples_of_sigsq.size(); n_g++){			
			sigsqs_to_export[n_g] = gibbs_samples_of_sigsq.get(n_g) * (TRANSFORM_Y ? y_range_sq : 1);			
		}
		return sigsqs_to_export;
	}
	
	public double[] getMuValuesForAllItersByTreeAndLeaf(int t, int leaf_num){
		double[] mu_vals = new double[num_gibbs_total_iterations];
		for (int n_g = 0; n_g < num_gibbs_total_iterations; n_g++){
//			System.out.println("n_g: " + n_g + "length of tree vec: " + gibbs_samples_of_cgm_trees.get(n_g).size());
			CGMTreeNode tree = gibbs_samples_of_cgm_trees.get(n_g).get(t);
			
			Double pred_y = tree.get_pred_for_nth_leaf(leaf_num);
			System.out.println("t: " + t + " leaf: " + leaf_num + " pred_y: " + pred_y);
			mu_vals[n_g] = un_transform_y(pred_y);
		}
		return mu_vals;
	}	
	
	private int maximalTreeGeneration(){
		int max_gen = Integer.MIN_VALUE;
		for (ArrayList<CGMTreeNode> cgm_trees : gibbs_samples_of_cgm_trees){
			if (cgm_trees != null){				
				for (CGMTreeNode tree : cgm_trees){					
					int gen = tree.deepestNode();
					if (gen >= max_gen){
						max_gen = gen;
					}
				}
			}
		}
		return max_gen;
	}
	
	public int maximalNodeNumber(){screwed up
		int max_gen = maximalTreeGeneration();
		int node_num = 0;
		for (int g = 0; g < max_gen; g++){
			node_num += (int)Math.pow(g, 2);
		}
		return node_num + 1;
	}
	
	public double[] getLikForTree(int t){
		return all_tree_liks[t];
	}
	
	public int[] getNumNodesForTreesInGibbsSamp(int n_g){
		ArrayList<CGMTreeNode> trees = gibbs_samples_of_cgm_trees.get(n_g);
		int[] num_nodes_by_tree = new int[trees.size()];
		for (int t = 0; t < trees.size(); t++){
			num_nodes_by_tree[t] = trees.get(t).numLeaves();
		}
		return num_nodes_by_tree;
	}	
	
	public int[] getDepthsForTreesInGibbsSamp(int n_g){
		ArrayList<CGMTreeNode> trees = gibbs_samples_of_cgm_trees.get(n_g);
		int[] depth_by_tree = new int[trees.size()];
		for (int t = 0; t < trees.size(); t++){
			depth_by_tree[t] = trees.get(t).deepestNode();
		}
		return depth_by_tree;
	}
	
	public String getRootSplits(int n_g){
		ArrayList<CGMTreeNode> trees = gibbs_samples_of_cgm_trees.get(n_g);
		ArrayList<String> root_splits = new ArrayList<String>(trees.size());
		for (int t = 0; t < trees.size(); t++){
			root_splits.add(trees.get(t).splitToString());
		}
		return IOTools.StringJoin(root_splits, "   ||   ");		
	}
	
	protected static void CloseDebugFiles(){
		tree_liks.close();
		remainings.close();
		sigsqs.close();
		sigsqs_draws.close();
		evaluations.close();
		other_debug.close();
	}
	
	protected static void OpenDebugFiles(){		
		try {
			sigsqs = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "sigsqs" + DEBUG_EXT, true)));
			other_debug = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "other_debug" + DEBUG_EXT, true)));
			sigsqs_draws = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "sigsqs_draws" + DEBUG_EXT, true)));
			tree_liks = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "tree_liks" + DEBUG_EXT, true)));
			evaluations = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "evaluations" + DEBUG_EXT, true)));
			remainings = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "remainings" + DEBUG_EXT, true)));	
		} catch (IOException e) {
			e.printStackTrace();
		}			
	}

	protected abstract CGMTreeNode SampleTreeByCalculatingRemainingsAndDrawingFromTreeDist(int i, int t, TreeArrayIllustration tree_array_illustration);

	protected double sampleInitialSigsqByDrawingFromThePrior() {
		//we're sampling from sigsq ~ InvGamma(nu / 2, nu * lambda / 2)
		//which is equivalent to sampling (1 / sigsq) ~ Gamma(nu / 2, 2 / (nu * lambda))
		return StatToolbox.sample_from_inv_gamma(hyper_nu / 2, 2 / (hyper_nu * hyper_lambda)); 
	}
	
	protected void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(CGMTreeNode node, double sigsq) {
//		System.out.println("assignLeafValsUsingPosteriorMeanAndCurrentSigsq sigsq: " + sigsq);
		if (node.isLeaf){
			double posterior_sigsq = calcLeafPosteriorVar(node, sigsq);
			//draw from posterior distribution
			double posterior_mean = calcLeafPosteriorMean(node, sigsq, posterior_sigsq);
//			System.out.println("posterior_mean = " + posterior_mean + " node.avg_response = " + node.avg_response());
			node.y_prediction = StatToolbox.sample_from_norm_dist(posterior_mean, posterior_sigsq);
			if (node.y_prediction == StatToolbox.ILLEGAL_FLAG){				
				node.y_prediction = 0.0; //this could happen on an empty node
//				System.out.println("ERROR assignLeafFINAL " + node.y_prediction + " (sigsq = " + sigsq + ")");
			}
			System.out.println("assignLeafFINAL " + un_transform_y(node.y_prediction) + " (sigsq = " + sigsq * y_range_sq + ")");
		}
		else {
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(node.left, sigsq);
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(node.right, sigsq);
		}
	}

	private double calcLeafPosteriorMean(CGMTreeNode node, double sigsq, double posterior_var) {
//		System.out.println("leafPosteriorMean hyper_sigsq_mu " + hyper_sigsq_mu + " node.n " + node.n + " sigsq " + sigsq + " node.avg_response() " + node.avg_response() + " posterior_var " + posterior_var);
		return (hyper_mu_mu / hyper_sigsq_mu + node.n / sigsq * node.avg_response()) / (1 / posterior_var);
	}

	private double calcLeafPosteriorVar(CGMTreeNode node, double sigsq) {
//		System.out.println("leafPosteriorVar sigsq " + sigsq + " var " + 1 / (1 / hyper_sigsq_mu + node.n * m / sigsq));
		return 1 / (1 / hyper_sigsq_mu + node.n / sigsq);
	}


	protected double samplePosteriorSigsq(int sample_num) {
		System.out.println("max tree node number: " + maximalNodeNumber());
		//first calculate the SSE
		double sum_sq_errors = 0;
		double[] es = getErrorsForAllTrees(sample_num);
		if (WRITE_DETAILED_DEBUG_FILES){		
			remainings.println((sample_num) + ",,e," + IOTools.StringJoin(es, ","));
			evaluations.println((sample_num) + ",,e," + IOTools.StringJoin(es, ","));
		}
		for (double e : es){
			sum_sq_errors += Math.pow(e, 2); 
		}
//		System.out.println("sample: " + sample_num + " sse = " + sum_sq_errors);
		//we're sampling from sigsq ~ InvGamma((nu + n) / 2, 1/2 * (sum_i error^2_i + lambda * nu))
		//which is equivalent to sampling (1 / sigsq) ~ Gamma((nu + n) / 2, 2 / (sum_i error^2_i + lambda * nu))
		double sigsq = StatToolbox.sample_from_inv_gamma((hyper_nu + n) / 2, 2 / (sum_sq_errors + hyper_nu * hyper_lambda));
//		System.out.println("\n\nSample posterior from trees " + treeIDsInCurrentSample(sample_num) + "  SSE=" + sum_sq_errors + " post_sigsq_sample: " + sigsq);
//		
//		//debug
//		double[] posterior_sigma_simus = new double[1000];
//		for (int i = 0; i < 1000; i++){
//			posterior_sigma_simus[i] = StatToolbox.sample_from_inv_gamma((hyper_nu + n) / 2, 2 / (sum_sq_errors + hyper_nu * hyper_lambda)) * (TRANSFORM_Y ? y_range_sq : 1);
//		}
//		System.out.print("\n\n\n");
//		sigsqs_draws.println(sample_num + "," + hyper_nu + "," + hyper_lambda + "," + n + "," + sum_sq_errors + "," + (sigsq * (TRANSFORM_Y ? y_range_sq : 1)) + "," + y_range_sq + "," + IOTools.StringJoin(posterior_sigma_simus, ","));
		
		return sigsq;
	}
	
//	private String treeIDsInCurrentSample(int sample_num){
//		ArrayList<CGMTreeNode> trees = gibbs_samples_of_cgm_trees.get(sample_num);
//		ArrayList<String> treeIds = new ArrayList<String>(trees.size());
//		for (int t = 0; t < trees.size(); t++){
//			treeIds.add(trees.get(t).stringID());
//		}
//		return IOTools.StringJoin(treeIds, ",");
//	}

	// hist(1 / rgamma(5000, 1.5, 1.5 * 153.65), br=100)
	protected void calculateHyperparameters() {
//		System.out.println("calculateHyperparameters in BART\n\n");
		double k = 2; //StatToolbox.inv_norm_dist(1 - (1 - CGMShared.MostOfTheDistribution) / 2.0);	
//		y_min = StatToolbox.sample_minimum(y);
//		y_max = StatToolbox.sample_maximum(y);
		if (TRANSFORM_Y){
			hyper_mu_mu = 0;
			hyper_sigsq_mu = Math.pow(YminAndYmaxHalfDiff / (k * Math.sqrt(m)), 2);
		}
		else {
			hyper_mu_mu = (y_min + y_max) / (2 * m); //ie the "center" of the distribution
			hyper_sigsq_mu = Math.pow((y_max - hyper_mu_mu) / (k * Math.sqrt(m)), 2); //margin of error over confidence spread with a correction factor for num trees
		}
		
		//first calculate \sigma_\mu
		
		//we fix nu and q		
		hyper_nu = 3.0;
		
		//now we do a simple search for the best value of lambda
		//if sig_sq ~ \nu\lambda * X where X is Inv chi sq, then sigsq ~ InvGamma(\nu/2, \nu\lambda/2) \neq InvChisq
		double s_sq_y = StatToolbox.sample_variance(y_trans);
//		double prob_diff = Double.MAX_VALUE;
		
//		double q = 0.9;
		double ten_pctile_chisq_df_3 = 0.5843744; //we need q=0.9 for this to work
		
		hyper_lambda = ten_pctile_chisq_df_3 / 3 * s_sq_y;
//		System.out.println("lambda: " + lambda);
//		
//		for (lambda = 0.00001; lambda < 10 * s_sq_y; lambda += (s_sq_y / 10000)){
//			double p = StatToolbox.cumul_dens_function_inv_gamma(hyper_nu / 2, hyper_nu * lambda / 2, s_sq_y);			
//			if (Math.abs(p - q) < prob_diff){
////				System.out.println("hyper_lambda = " + hyper_lambda + " lambda = " + lambda + " p = " + p + " ssq = " + ssq);
//				hyper_lambda = lambda;
//				prob_diff = Math.abs(p - q);
//			}
//		}
		System.out.println("y_min = " + y_min + " y_max = " + y_max + " R_y = " + Math.sqrt(y_range_sq));
		System.out.println("hyperparams:  k = " + k + " hyper_mu_mu = " + hyper_mu_mu + " sigsq_mu = " + hyper_sigsq_mu + " hyper_lambda = " + hyper_lambda + " hyper_nu = " + hyper_nu + " s_y_trans^2 = " + s_sq_y + " R_y = " + Math.sqrt(y_range_sq));
	}

	
	/**
	 * We call then override the setData function. we need to ensure y is properly transformed pursuant to 
	 * bottom of p271
	 */
//	public void setData(ArrayList<double[]> X_y){
//		y_trans = y; //do this first and let it be overwritten by the super function
//		super.setData(X_y);
		
		//DEBUG STUFF
//		ArrayList<String> y_header = new ArrayList<String>(n);
//		for (int i = 0; i < n; i++){
//			y_header.add("y_" + i);
//		}
//
//		remainings.println("iter,tree_num_left_out,tree_id," + IOTools.StringJoin(y_header, ","));		
//		evaluations.println("iter,tree_num,tree_id," + IOTools.StringJoin(y_header, ","));		
//	}
	
	//make sure you get the prior correct if you don't transform
	protected static final double YminAndYmaxHalfDiff = 0.5;
	protected void transformResponseVariable() {
		System.out.println("CGMBART transformResponseVariable");
		//make sure to initialize the y_trans to be y first
		super.transformResponseVariable();
		//make data we need later
		y_min = StatToolbox.sample_minimum(y);
		y_max = StatToolbox.sample_maximum(y);
		y_range_sq = Math.pow(y_max - y_min, 2);

		if (TRANSFORM_Y){
			for (int i = 0; i < n; i++){
				y_trans[i] = (y[i] - y_min) / (y_max - y_min) - YminAndYmaxHalfDiff;
			}
		}
		//debug stuff
//		y_and_y_trans.println("y,y_trans");
//		for (int i = 0; i < n; i++){
//			System.out.println("y_trans[i] = " + y_trans[i] + " y[i] = " + y[i] + " y_untransform = " + un_transform_y(y_trans[i]));
//			y_and_y_trans.println(y[i] + "," + y_trans[i]);
//		}
//		y_and_y_trans.close();
	}

	protected double un_transform_y(Double yt_i){
		if (yt_i == null){
			return -9999999;
		}
		if (TRANSFORM_Y){
			return (yt_i + YminAndYmaxHalfDiff) * (y_max - y_min) + y_min;
		}
		else {
			return yt_i;
		}
	}	
	
	protected double[] getResidualsForAllTreesExcept(int j, int sample_num){		
		double[] sum_ys_without_jth_tree = new double[n];
		ArrayList<CGMTreeNode> trees = gibbs_samples_of_cgm_trees.get(sample_num);

		for (int i = 0; i < n; i++){
			sum_ys_without_jth_tree[i] = 0; //initialize at zero, then add it up over all trees except the jth
			for (int t = 0; t < trees.size(); t++){
				if (t != j){
					//obviously y_vec - \sum_i g_i = \sum_i y_i - g_i
					sum_ys_without_jth_tree[i] += trees.get(t).Evaluate(X_y.get(i)); //first tree for now
				}
			}
		}
		//now we need to subtract this from y
		double[] Rjs = new double[n];
		for (int i = 0; i < n; i++){
			Rjs[i] = y_trans[i] - sum_ys_without_jth_tree[i];
		}
//		System.out.println("getResidualsForAllTreesExcept " + (j+1) + "th tree:  " +  new DoubleMatrix(Rjs).transpose().toString(2));
		return Rjs;
	}
	
	private double[] getErrorsForAllTrees(int sample_num){
//		System.out.println("getErrorsForAllTrees");
		double[] sum_ys = new double[n];
		for (int i = 0; i < n; i++){
			sum_ys[i] = 0; //initialize at zero, then add it up over all trees except the jth
			for (int t = 0; t < m; t++){
//				System.out.println("getErrorsForAllTrees m = " + m);
				//obviously y_vec - \sum_i g_i = \sum_i y_i - g_i
				CGMTreeNode tree = gibbs_samples_of_cgm_trees.get(sample_num).get(t);
				double y_hat = tree.Evaluate(X_y.get(i));
//				System.out.println("i = " + (i + 1) + " y: " + y_trans[i] + " y_hat: " + y_hat + " e: " + (y_trans[i] - y_hat)+ " tree " + tree.stringID());
				sum_ys[i] += y_hat;
			}
		}
		//now we need to subtract this from y
		double[] errorjs = new double[n];
		for (int i = 0; i < n; i++){
			errorjs[i] = y_trans[i] - sum_ys[i];
		}
//		System.out.println("sum_ys " + IOTools.StringJoin(sum_ys, ","));
//		System.out.println("y_trans " + IOTools.StringJoin(y_trans, ","));
//		System.out.println("errorjs " + IOTools.StringJoin(errorjs, ","));
		return errorjs;
	}	


	@Override
	protected void FlushData() {
		for (ArrayList<CGMTreeNode> cgm_trees : gibbs_samples_of_cgm_trees){
			for (CGMTreeNode tree : cgm_trees){
				tree.flushNodeData();
			}
		}	
	}

	@Override
	public void StopBuilding() {
		stop_bit = true;
	}

}