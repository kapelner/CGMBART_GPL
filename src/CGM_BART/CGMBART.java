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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import CGM_Statistics.*;

@SuppressWarnings("serial")
public abstract class CGMBART extends CGMBART_debug implements Serializable  {


	public CGMBART(){}
	
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);		
		//this posterior builder will be shared throughout the entire process
		posterior_builder = new CGMBARTPosteriorBuilder(this);
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

	
	protected void SetupGibbsSampling(){
		all_tree_liks = new double[num_trees][num_gibbs_total_iterations + 1];

		//now initialize the gibbs sampler array for trees and error variances
		gibbs_samples_of_cgm_trees = new ArrayList<ArrayList<CGMBARTTreeNode>>(num_gibbs_total_iterations);
		gibbs_samples_of_cgm_trees_after_burn_in = new ArrayList<ArrayList<CGMBARTTreeNode>>(num_gibbs_total_iterations - num_gibbs_burn_in);
		gibbs_samples_of_cgm_trees.add(null);
		gibbs_samples_of_sigsq = new ArrayList<Double>(num_gibbs_total_iterations);	
		gibbs_samples_of_sigsq_after_burn_in = new ArrayList<Double>(num_gibbs_total_iterations - num_gibbs_burn_in);
		
		InitizializeSigsq();
		InitiatizeTrees();
		InitializeMus();		
		DebugInitialization();		
	}
	
	protected void InitiatizeTrees() {
		ArrayList<CGMBARTTreeNode> cgm_trees = new ArrayList<CGMBARTTreeNode>(num_trees);
		//now we're going to build each tree based on the prior given in section 2 of the paper
		//first thing is first, we create the tree structures using priors for p(T_1), p(T_2), .., p(T_m)
		
		for (int i = 0; i < num_trees; i++){
//			System.out.println("CGMBART create prior on tree: " + (i + 1));
			CGMBARTTreeNode tree = new CGMBARTTreeNode(null, X_y, this);
			tree.y_prediction = 0.0; //default
//			modifyTreeForDebugging(tree);
			tree.updateWithNewResponsesAndPropagate(X_y, y_trans, p);
			cgm_trees.add(tree);
		}	
		gibbs_samples_of_cgm_trees.add(0, cgm_trees);		
	}

	protected void InitializeMus() {
		for (CGMBARTTreeNode tree : gibbs_samples_of_cgm_trees.get(0)){
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

	protected void DoGibbsSampling(){	
		for (gibb_sample_i = 1; gibb_sample_i <= num_gibbs_total_iterations; gibb_sample_i++){
			tree_liks.print(gibb_sample_i + ",");
			final ArrayList<CGMBARTTreeNode> cgm_trees = new ArrayList<CGMBARTTreeNode>(num_trees);				
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

	private void FlushTempDataForSample(ArrayList<CGMBARTTreeNode> cgm_trees) {
		for (CGMBARTTreeNode tree : cgm_trees){
			tree.flushNodeData();	
		}
	}



	protected void SampleSigsq(int sample_num) {
		double sigsq = drawSigsqFromPosterior(sample_num);
		gibbs_samples_of_sigsq.add(sample_num, sigsq);
		posterior_builder.setCurrentSigsqValue(sigsq);
	}

	protected void SampleMus(int sample_num, int t) {
//		System.out.println("SampleMu sample_num " +  sample_num + " t " + t + " gibbs array " + gibbs_samples_of_cgm_trees.get(sample_num));
		CGMBARTTreeNode tree = gibbs_samples_of_cgm_trees.get(sample_num).get(t);
		assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(tree, gibbs_samples_of_sigsq.get(sample_num - 1));
	}

	protected void SampleTree(int sample_num, int t, ArrayList<CGMBARTTreeNode> cgm_trees, TreeArrayIllustration tree_array_illustration) {
		
		final CGMBARTTreeNode copy_of_old_jth_tree = gibbs_samples_of_cgm_trees.get(sample_num - 1).get(t).clone(true);
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
		
		ArrayList<CGMBARTTreeNode> other_trees = new ArrayList<CGMBARTTreeNode>(num_trees - 1);
		for (int j = 0; j < t; j++){
			other_trees.add(gibbs_samples_of_cgm_trees.get(sample_num).get(j));
		}
		for (int j = t + 1; j < num_trees; j++){
			other_trees.add(gibbs_samples_of_cgm_trees.get(sample_num - 1).get(j));
		}		
		
		final double[] R_j = getResidualsBySubtractingTrees(other_trees);
		
//		System.out.println("SampleTreeByDrawingFromTreeDist rs = " + IOTools.StringJoin(R_j, ","));
		if (WRITE_DETAILED_DEBUG_FILES){
			remainings.println((sample_num - 1) + "," + t + "," + copy_of_old_jth_tree.stringID() + "," + Tools.StringJoin(R_j, ","));			
		}
		
		//now, (important!) set the R_j's as this tree's data.
		copy_of_old_jth_tree.updateWithNewResponsesAndPropagate(X_y, R_j, p);
		
		//sample from T_j | R_j, \sigma
		//now we will run one M-H step on this tree with the y as the R_j
		CGMBARTTreeNode tree_star = posterior_builder.iterateMHPosteriorTreeSpaceSearch(copy_of_old_jth_tree);
		
		//DEBUG
//		System.err.println("tree star: " + tree_star.stringID() + " tree num leaves: " + tree_star.numLeaves() + " tree depth:" + tree_star.deepestNode());
//		double lik = tree_star.
//		tree_liks.print(lik + "," + tree_star.stringID() + ",");
//		tree_array_illustration.addLikelihood(lik);
//		all_tree_liks[t][sample_num] = lik;
		
		cgm_trees.add(tree_star);
		//now set the new trees in the gibbs sample pantheon, keep updating it...
		gibbs_samples_of_cgm_trees.set(sample_num, cgm_trees);
//		System.out.println("SampleTree sample_num " + sample_num + " cgm_trees " + cgm_trees);
		
		tree_array_illustration.AddTree(tree_star);
	}	


	
	private double[] getResidualsBySubtractingTrees(ArrayList<CGMBARTTreeNode> other_trees) {
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
	


	protected double[] getGibbsSamplesForPrediction(double[] record){
//		System.out.println("eval record: " + record + " numtrees:" + this.bayesian_trees.size());
		//the results for each of the gibbs samples
		double[] y_gibbs_samples = new double[numSamplesAfterBurningAndThinning()];	
		for (int i = 0; i < numSamplesAfterBurningAndThinning(); i++){
			ArrayList<CGMBARTTreeNode> cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in.get(i);
			double yt_i = 0;
			for (CGMBARTTreeNode tree : cgm_trees){ //sum of trees right?
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

	


	protected double sampleInitialSigsqByDrawingFromThePrior() {
		//we're sampling from sigsq ~ InvGamma(nu / 2, nu * lambda / 2)
		//which is equivalent to sampling (1 / sigsq) ~ Gamma(nu / 2, 2 / (nu * lambda))
		return StatToolbox.sample_from_inv_gamma(hyper_nu / 2, 2 / (hyper_nu * hyper_lambda)); 
	}
	
	protected void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(CGMBARTTreeNode node, double sigsq) {
//		System.out.println("assignLeafValsUsingPosteriorMeanAndCurrentSigsq sigsq: " + sigsq);
		if (node.isLeaf){
			double posterior_sigsq = calcLeafPosteriorVar(node, sigsq);
			//draw from posterior distribution
			double posterior_mean = calcLeafPosteriorMean(node, sigsq, posterior_sigsq);
//			System.out.println("posterior_mean = " + posterior_mean + " node.avg_response = " + node.avg_response());
			node.y_prediction = StatToolbox.sample_from_norm_dist(posterior_mean, posterior_sigsq);
			if (node.y_prediction == StatToolbox.ILLEGAL_FLAG){				
				node.y_prediction = 0.0; //this could happen on an empty node
				System.err.println("ERROR assignLeafFINAL " + node.y_prediction + " (sigsq = " + sigsq + ")");
			}
//			System.out.println("assignLeafFINAL " + un_transform_y(node.y_prediction) + " (sigsq = " + sigsq * y_range_sq + ")");
		}
		else {
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(node.left, sigsq);
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(node.right, sigsq);
		}
	}

	private double calcLeafPosteriorMean(CGMBARTTreeNode node, double sigsq, double posterior_var) {
//		System.out.println("leafPosteriorMean hyper_sigsq_mu " + hyper_sigsq_mu + " node.n " + node.n + " sigsq " + sigsq + " node.avg_response() " + node.avg_response() + " posterior_var " + posterior_var);
		return (hyper_mu_mu / hyper_sigsq_mu + node.n_at_this_juncture / sigsq * node.avgResponse()) / (1 / posterior_var);
	}

	private double calcLeafPosteriorVar(CGMBARTTreeNode node, double sigsq) {
//		System.out.println("leafPosteriorVar sigsq " + sigsq + " var " + 1 / (1 / hyper_sigsq_mu + node.n * m / sigsq));
		return 1 / (1 / hyper_sigsq_mu + node.n_at_this_juncture / sigsq);
	}


	protected double drawSigsqFromPosterior(int sample_num) {
		//first calculate the SSE
		double sum_sq_errors = 0;
		double[] es = getErrorsForAllTrees(sample_num);
		if (WRITE_DETAILED_DEBUG_FILES){		
			remainings.println((sample_num) + ",,e," + Tools.StringJoin(es, ","));
			evaluations.println((sample_num) + ",,e," + Tools.StringJoin(es, ","));
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
	


	
	
	protected double[] getResidualsForAllTreesExcept(int j, int sample_num){		
		double[] sum_ys_without_jth_tree = new double[n];
		ArrayList<CGMBARTTreeNode> trees = gibbs_samples_of_cgm_trees.get(sample_num);

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
				CGMBARTTreeNode tree = gibbs_samples_of_cgm_trees.get(sample_num).get(t);
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

}