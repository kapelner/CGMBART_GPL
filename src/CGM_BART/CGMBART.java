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
	protected static final int DEFAULT_NUM_GIBBS_BURN_IN = 500;
	protected static final int DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS = 2000; //this must be larger than the number of burn in!!!
	
	protected static PrintWriter y_and_y_trans;
	protected static PrintWriter sigsqs;
	protected static PrintWriter other_debug;
	protected static PrintWriter sigsqs_draws;
	protected static PrintWriter tree_liks;
	protected static PrintWriter remainings;
	protected static PrintWriter evaluations;
	public static PrintWriter mh_iterations_full_record;
	
	protected static boolean TREE_ILLUST = true;
	protected static final boolean WRITE_DETAILED_DEBUG_FILES = false;
	
	protected static final String CSVFileFromRName = "bart_data.csv";
	protected static final String CSVFileFromRDirectory = "datasets";	
	
//	static {
//		Classifier.writeToDebugLog();
//	}

	static {
		try {			
			output = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "output" + DEBUG_EXT)));
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
			mh_iterations_full_record = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "mh_iterations_full_record" + DEBUG_EXT)));
			mh_iterations_full_record.println(
					"step" + "," + 
					"node_to_change" + "," + 
					"loc" + "," +
					"a_i" + "," +
					"v_i" + "," +
					"a_*" + "," +	
					"v_*" + "," +
					"leaf_1_*" + "," + 
					"leaf_2_*" + "," + 
					"leaf_3_*" + "," + 
					"leaf_4_*" + "," + 
					"tree_*_likelihood" + "," + 
					"leaf_1_i" + "," + 
					"leaf_2_i" + "," + 
					"leaf_3_i" + "," + 
					"leaf_4_i" + "," + 
					"tree_i_likelihood" + "," + 	
					"accept_or_reject" + "," + 
					"ln_r" + "," +
					"ln_u_0_1"
				);			
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
	protected int num_trees;
	protected int num_gibbs_burn_in;
	protected int num_gibbs_total_iterations;	
	/** all the hyperparameters */
	protected double hyper_mu_mu;
	protected double hyper_sigsq_mu;
	protected double hyper_nu;
	protected double hyper_lambda;
	/** we will use the tree prior builder to initally build the trees and later in the gibbs sampler as well */
	protected CGMBARTPriorBuilder tree_prior_builder;
	/** we will use the tree posterior builder to calculated liks and do the M-H step */
	protected CGMBARTPosteriorBuilder posterior_builder;
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
		num_trees = DEFAULT_NUM_TREES;
		num_gibbs_burn_in = DEFAULT_NUM_GIBBS_BURN_IN;
		num_gibbs_total_iterations = DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS;
	}	
	
	public void setNumTrees(int m){
		this.num_trees = m;
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
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		//do things that can be done as soon as the data is known
		
		//first establish the hyperparams sigsq_mu, nu, lambda
		calculateHyperparameters();	
		//now generate a prior builder... used in any implementation
		tree_prior_builder = new CGMBARTPriorBuilder(X_y, p);
		//this posterior builder will be shared throughout the entire process
		posterior_builder = new CGMBARTPosteriorBuilder(tree_prior_builder);
		//we have to set the CGM98 hyperparameters as well as the hyperparameter sigsq_mu
		posterior_builder.setHyperparameters(hyper_mu_mu, hyper_sigsq_mu);	
	}

	@Override
	public void Build() {
		//this can be different for any BART implementation
		SetupGibbsSampling();
		//this section is different for the different BART implementations
		//but it basically does all the Gibbs sampling
		DoGibbsSampling();
		//now we burn and thin the chains for each param
		BurnTreeAndSigsqChain();
		//for some reason we don't thin... I don't really understand why... has to do with autocorrelation and stickiness?
//		ThinBothChains();
		//make sure debug files are closed
		CloseDebugFiles();
	}
	
	public int currentGibbsSampleIteration(){
		return gibb_sample_i;
	}
	
	public boolean gibbsFinished(){
		return gibb_sample_i >= num_gibbs_total_iterations;
	}
	
	protected void SetupGibbsSampling(){
		all_tree_liks = new double[num_trees][num_gibbs_total_iterations + 1];

		//now initialize the gibbs sampler array for trees and error variances
		gibbs_samples_of_cgm_trees = new ArrayList<ArrayList<CGMTreeNode>>(num_gibbs_total_iterations);
		gibbs_samples_of_cgm_trees_after_burn_in = new ArrayList<ArrayList<CGMTreeNode>>(num_gibbs_total_iterations - num_gibbs_burn_in);
		gibbs_samples_of_cgm_trees.add(null);
		gibbs_samples_of_sigsq = new ArrayList<Double>(num_gibbs_total_iterations);	
		gibbs_samples_of_sigsq_after_burn_in = new ArrayList<Double>(num_gibbs_total_iterations - num_gibbs_burn_in);
		
		InitizializeSigsq();
		InitiatizeTrees();
		InitializeMus();		
		DebugInitialization();		
	}

	protected void DoGibbsSampling(){	
		for (gibb_sample_i = 1; gibb_sample_i <= num_gibbs_total_iterations; gibb_sample_i++){
			tree_liks.print(gibb_sample_i + ",");
			final ArrayList<CGMTreeNode> cgm_trees = new ArrayList<CGMTreeNode>(num_trees);				
			final TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(gibb_sample_i);			
			//we cycle over each tree and update it according to formulas 15, 16 on p274
			for (int t = 0; t < num_trees; t++){
				if (t == 0){
					System.out.println("Sampling M_" + (t + 1) + "/" + num_trees + " iter " + gibb_sample_i + "/" + num_gibbs_total_iterations);
				}				
				SampleTree(gibb_sample_i, t, cgm_trees, tree_array_illustration);
				SampleMus(gibb_sample_i, t);
				gibbs_samples_of_cgm_trees.add(gibb_sample_i, cgm_trees);
				if (stop_bit){
					return;
				}				
			}
			SampleSigsq(gibb_sample_i);
			DebugSample(gibb_sample_i, tree_array_illustration);
			FlushTempDataForSample(cgm_trees);
			
			if (PrintOutEvery != null && gibb_sample_i % PrintOutEvery == 0){
				System.out.println("gibbs iter: " + gibb_sample_i + "/" + num_gibbs_total_iterations);
			}
		}
	}

	private void FlushTempDataForSample(ArrayList<CGMTreeNode> cgm_trees) {
		for (CGMTreeNode tree : cgm_trees){
			tree.flushNodeData();	
		}
	}

	protected void DebugSample(int sample_num, TreeArrayIllustration tree_array_illustration) {

		if (WRITE_DETAILED_DEBUG_FILES){	
			remainings.println((sample_num) + ",,y," + IOTools.StringJoin(y_trans, ","));
			
			ArrayList<CGMTreeNode> current_trees = gibbs_samples_of_cgm_trees.get(sample_num);
			for (int t = 0; t < num_trees; t++){
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
//		if (StatToolbox.rand() < 0.0333){
			if (TREE_ILLUST){
				tree_array_illustration.CreateIllustrationAndSaveImage();
			}
//		}
//			}
//		};
//		illustrator_thread.start();
		
		tree_liks.print("\n");	
		
		if (TRANSFORM_Y){
			sigsqs.println(sample_num + "," + gibbs_samples_of_sigsq.get(sample_num) * y_range_sq);	
		}
		else {
			sigsqs.println(sample_num + "," + gibbs_samples_of_sigsq.get(sample_num));
		}

		//now close and open all debug
		if (StatToolbox.rand() < 0.0333){
			CloseDebugFiles();
			OpenDebugFiles();
		}
	}

	protected void SampleSigsq(int sample_num) {
		double sigsq = drawSigsqFromPosterior(sample_num);
		gibbs_samples_of_sigsq.add(sample_num, sigsq);
		posterior_builder.setCurrentSigsqValue(sigsq);
	}

	protected void SampleMus(int sample_num, int t) {
//		System.out.println("SampleMu sample_num " +  sample_num + " t " + t + " gibbs array " + gibbs_samples_of_cgm_trees.get(sample_num));
		CGMTreeNode tree = gibbs_samples_of_cgm_trees.get(sample_num).get(t);
		assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(tree, gibbs_samples_of_sigsq.get(sample_num - 1));
	}

	protected void SampleTree(int sample_num, int t, ArrayList<CGMTreeNode> cgm_trees, TreeArrayIllustration tree_array_illustration) {
		
		final CGMTreeNode copy_of_old_jth_tree = gibbs_samples_of_cgm_trees.get(sample_num - 1).get(t).clone(true);
//		System.out.println("copy_of_old_jth_tree.data:" + copy_of_old_jth_tree.data + "\n orig_tree.data:" + gibbs_samples_of_cgm_trees.get(sample_num - 1).get(t).data);
//		System.out.println("SampleTreeByCalculatingRemainingsAndDrawingFromTreeDist t:" + t + " of m:" + m);
//		ArrayList<CGMTreeNode> leaves = tree.getTerminalNodes();
//		for (int b = 0; b < leaves.size(); b++){
//			CGMTreeNode leaf = leaves.get(b);
//			DoubleMatrix rs = new DoubleMatrix(leaf.get_ys_in_data());
//			System.out.println("tree " + tree.stringID() + " leaf " + b + " ys:\n" + rs.transpose().toString(2));
//			
//		}
		//okay so first we need to get "y" that this tree sees. This is defined as R_j
		//in formula 12 on p274
		
		//who are the previous trees. 
		//e.g. if t=0, then we take all 1, ..., m-1 trees from previous gibbs sample
		//     if t=1, then we take the 0th tree from this gibbs sample, and 2, ..., m-1 trees from the previous gibbs sample
		//     ...
		//     if t=j, then we take the 0,...,j-1 trees from this gibbs sample, and j+1, ..., m-1 trees from the previous gibbs sample
		//     ...
		//     if t=m-1, then we take all 0, ..., m-2 trees from this gibbs sample
		// so let's put together this list of trees:
		
		ArrayList<CGMTreeNode> other_trees = new ArrayList<CGMTreeNode>(num_trees - 1);
		for (int j = 0; j < t; j++){
			other_trees.add(gibbs_samples_of_cgm_trees.get(sample_num).get(j));
		}
		for (int j = t + 1; j < num_trees; j++){
			other_trees.add(gibbs_samples_of_cgm_trees.get(sample_num - 1).get(j));
		}		
		
		final double[] R_j = getResidualsBySubtractingTrees(other_trees);
		
//		System.out.println("SampleTreeByDrawingFromTreeDist rs = " + IOTools.StringJoin(R_j, ","));
		if (WRITE_DETAILED_DEBUG_FILES){
			remainings.println((sample_num - 1) + "," + t + "," + copy_of_old_jth_tree.stringID() + "," + IOTools.StringJoin(R_j, ","));			
		}
		
		//now, (important!) set the R_j's as this tree's data.
		copy_of_old_jth_tree.updateWithNewResponsesAndPropagate(X_y, R_j, p);
		
		//sample from T_j | R_j, \sigma
		//now we will run one M-H step on this tree with the y as the R_j
		CGMTreeNode tree_star = posterior_builder.iterateMHPosteriorTreeSpaceSearch(copy_of_old_jth_tree);
		
		//DEBUG
//		System.err.println("tree star: " + tree_star.stringID() + " tree num leaves: " + tree_star.numLeaves() + " tree depth:" + tree_star.deepestNode());
		double lik = tree_star.log_prop_lik;
		tree_liks.print(lik + "," + tree_star.stringID() + ",");
		tree_array_illustration.addLikelihood(lik);
		all_tree_liks[t][sample_num] = lik;
		
		cgm_trees.add(tree_star);
		//now set the new trees in the gibbs sample pantheon, keep updating it...
		gibbs_samples_of_cgm_trees.set(sample_num, cgm_trees);
//		System.out.println("SampleTree sample_num " + sample_num + " cgm_trees " + cgm_trees);
		
		tree_array_illustration.AddTree(tree_star);
	}	

	protected void DebugInitialization() {
		ArrayList<CGMTreeNode> initial_trees = gibbs_samples_of_cgm_trees.get(0);
			
		if (TREE_ILLUST){
			TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(0);
			for (CGMTreeNode tree : initial_trees){
				tree_array_illustration.AddTree(tree);
				tree_array_illustration.addLikelihood(0);			
			}
			tree_array_illustration.CreateIllustrationAndSaveImage();
		}
		
		if (WRITE_DETAILED_DEBUG_FILES){
			for (int t = 0; t < num_trees; t++){
				CGMTreeNode tree = initial_trees.get(t);
				ArrayList<String> all_results = new ArrayList<String>(n);
				for (int i = 0; i < n; i++){
					all_results.add("" + tree.Evaluate(X_y.get(i))); //TreeIllustration.one_digit_format.format(
				} 
				evaluations.println(0 + "," + t + "," + tree.stringID() + "," + IOTools.StringJoin(all_results, ","));
			}
		}
	}
	
	private double[] getResidualsBySubtractingTrees(ArrayList<CGMTreeNode> other_trees) {
		double[] sum_ys_without_jth_tree = new double[n];

		for (int i = 0; i < n; i++){
			sum_ys_without_jth_tree[i] = 0; //initialize at zero, then add it up over all trees except the jth
			for (int t = 0; t < other_trees.size(); t++){
				sum_ys_without_jth_tree[i] += other_trees.get(t).Evaluate(X_y.get(i));
			}
		}
		//now we need to subtract this from y
		double[] Rjs = new double[n];
		for (int i = 0; i < n; i++){
			Rjs[i] = y_trans[i] - sum_ys_without_jth_tree[i];
		}
//		System.out.println("getResidualsForAllTreesExcept one " +  new DoubleMatrix(Rjs).transpose().toString(2));
		return Rjs;
	}		

	protected void InitiatizeTrees() {
		ArrayList<CGMTreeNode> cgm_trees = new ArrayList<CGMTreeNode>(num_trees);
		//now we're going to build each tree based on the prior given in section 2 of the paper
		//first thing is first, we create the tree structures using priors for p(T_1), p(T_2), .., p(T_m)
		
		for (int i = 0; i < num_trees; i++){
//			System.out.println("CGMBART create prior on tree: " + (i + 1));
			CGMTreeNode tree = new CGMTreeNode(null, X_y, this);
			tree.y_prediction = 0.0; //default
			tree.initLogPropLik();
//			modifyTreeForDebugging(tree);
			tree.updateWithNewResponsesAndPropagate(X_y, y_trans, p);
			cgm_trees.add(tree);
		}	
		gibbs_samples_of_cgm_trees.add(0, cgm_trees);		
	}

	protected void InitializeMus() {
		for (CGMTreeNode tree : gibbs_samples_of_cgm_trees.get(0)){
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(tree, gibbs_samples_of_sigsq.get(0));
		}		
	}

	protected void InitizializeSigsq() {
		gibbs_samples_of_sigsq.add(0, sampleInitialSigsqByDrawingFromThePrior());		
	}

	private void BurnTreeAndSigsqChain() {		
		for (int i = num_gibbs_burn_in; i < num_gibbs_total_iterations; i++){
			gibbs_samples_of_cgm_trees_after_burn_in.add(gibbs_samples_of_cgm_trees.get(i));
			gibbs_samples_of_sigsq_after_burn_in.add(gibbs_samples_of_sigsq.get(i));
		}	
		System.out.println("BurnTreeAndSigsqChain gibbs_samples_of_sigsq_after_burn_in length = " + gibbs_samples_of_sigsq_after_burn_in.size());
	}	
	
	
	/*
	 * 
	 * 
	 * 
	 * Everything that has to do with evaluation
	 * 
	 * 
	 */
	
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
	
	/*
	 * 
	 * 
	 * 
	 * Everything that has to do with evaluation
	 * 
	 * 
	 */
	
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
//			System.out.println("t: " + t + " leaf: " + leaf_num + " pred_y: " + pred_y);
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
	
	public int maximalNodeNumber(){
		int max_gen = maximalTreeGeneration();
		int node_num = 0;
		for (int g = 0; g <= max_gen; g++){
			node_num += (int)Math.pow(2, g);
		}
		return node_num;
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
		mh_iterations_full_record.close();
	}
	
	protected static void OpenDebugFiles(){		
		try {
			sigsqs = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "sigsqs" + DEBUG_EXT, true)));
			other_debug = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "other_debug" + DEBUG_EXT, true)));
			sigsqs_draws = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "sigsqs_draws" + DEBUG_EXT, true)));
			tree_liks = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "tree_liks" + DEBUG_EXT, true)));
			evaluations = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "evaluations" + DEBUG_EXT, true)));
			remainings = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "remainings" + DEBUG_EXT, true)));	
			mh_iterations_full_record = new PrintWriter(new BufferedWriter(new FileWriter(CGMShared.DEBUG_DIR + File.separatorChar + "mh_iterations_full_record" + DEBUG_EXT, true)));
			
		} catch (IOException e) {
			e.printStackTrace();
		}			
	}

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
				System.out.println("ERROR assignLeafFINAL " + node.y_prediction + " (sigsq = " + sigsq + ")");
			}
//			System.out.println("assignLeafFINAL " + un_transform_y(node.y_prediction) + " (sigsq = " + sigsq * y_range_sq + ")");
		}
		else {
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(node.left, sigsq);
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(node.right, sigsq);
		}
	}

	private double calcLeafPosteriorMean(CGMTreeNode node, double sigsq, double posterior_var) {
//		System.out.println("leafPosteriorMean hyper_sigsq_mu " + hyper_sigsq_mu + " node.n " + node.n + " sigsq " + sigsq + " node.avg_response() " + node.avg_response() + " posterior_var " + posterior_var);
		return (hyper_mu_mu / hyper_sigsq_mu + node.n / sigsq * node.avgResponse()) / (1 / posterior_var);
	}

	private double calcLeafPosteriorVar(CGMTreeNode node, double sigsq) {
//		System.out.println("leafPosteriorVar sigsq " + sigsq + " var " + 1 / (1 / hyper_sigsq_mu + node.n * m / sigsq));
		return 1 / (1 / hyper_sigsq_mu + node.n / sigsq);
	}


	protected double drawSigsqFromPosterior(int sample_num) {
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
			hyper_sigsq_mu = Math.pow(YminAndYmaxHalfDiff / (k * Math.sqrt(num_trees)), 2);
		}
		else {
			hyper_mu_mu = (y_min + y_max) / (2 * num_trees); //ie the "center" of the distribution
			hyper_sigsq_mu = Math.pow((y_max - hyper_mu_mu) / (k * Math.sqrt(num_trees)), 2); //margin of error over confidence spread with a correction factor for num trees
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
		System.out.println("hyperparams:  k = " + k + " hyper_mu_mu = " + hyper_mu_mu + " sigsq_mu = " + hyper_sigsq_mu + " hyper_lambda = " + hyper_lambda + " hyper_nu = " + hyper_nu + " s_y_trans^2 = " + s_sq_y + " R_y = " + Math.sqrt(y_range_sq) + "\n\n");
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
				y_trans[i] = transform_y(y[i]);
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
	
	protected double transform_y(double y_i){
		return (y_i - y_min) / (y_max - y_min) - YminAndYmaxHalfDiff;
	}
	
	public double un_transform_y(double yt_i){
		if (TRANSFORM_Y){
//			System.out.println("un_transform_y TRANSFORM_Y");
			return (yt_i + YminAndYmaxHalfDiff) * (y_max - y_min) + y_min;
		}
		else {
			return yt_i;
		}
	}

	public double un_transform_y(Double yt_i){
		if (yt_i == null){
			return -9999999;
		}
		return un_transform_y((double)yt_i);
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
			for (int t = 0; t < num_trees; t++){
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
	
	protected CGMTreeNode CreateTheSimpleTreeModel() {
		CGMTreeNode root = new CGMTreeNode(null, null, this);
		CGMTreeNode left = new CGMTreeNode(null, null, this);
		CGMTreeNode leftleft = new CGMTreeNode(null, null, this);
		CGMTreeNode leftright = new CGMTreeNode(null, null, this);
		CGMTreeNode right = new CGMTreeNode(null, null, this);
		CGMTreeNode rightleft = new CGMTreeNode(null, null, this);
		CGMTreeNode rightright = new CGMTreeNode(null, null, this);

		root.isLeaf = false;
		root.splitAttributeM = 0;
		root.splitValue = 30.0;
		root.left = left;
		root.right = right;	

		left.parent = root;
		left.isLeaf = false;
		left.splitAttributeM = 2;
		left.splitValue = 10.0;	
		left.left = leftleft;
		left.right = leftright;

		leftleft.parent = left;
		leftleft.isLeaf = true;
		leftleft.y_prediction = transform_y(10);		

		leftright.parent = left;
		leftright.isLeaf = true;
		leftright.y_prediction = transform_y(30);

		right.parent = root;
		right.isLeaf = false;
		right.splitAttributeM = 1;
		right.splitValue = 80.0;	
		right.left = rightleft;
		right.right = rightright;

		rightleft.parent = right;
		rightleft.isLeaf = true;
		rightleft.y_prediction = transform_y(50);		

		rightright.parent = right;
		rightright.isLeaf = true;
		rightright.y_prediction = transform_y(70);

		//make sure there's data in there
		root.updateWithNewResponsesAndPropagate(X_y, y_trans, p);
		return root;
	}	

}