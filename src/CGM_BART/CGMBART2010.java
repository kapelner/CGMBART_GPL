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

import java.util.ArrayList;

import CGM_Statistics.*;
import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentTools.IOTools;
import GemIdentView.JProgressBarAndLabel;

/**
 * This class is a faithful representation of the CGM 2010 paper
 * @author kapelner
 *
 */
@SuppressWarnings("serial")
public abstract class CGMBART2010 extends CGMBART {
	
	public CGMBART2010(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
	}

	protected void CoreBuild(){	
		InitiateGibbsChain();
		
		//now we're off to the races, gibbs sampling away. We begin at sample **1** since we've already built the zeroth sample from the prior
		for (gibb_sample_i = 1; gibb_sample_i <= num_gibbs_total_iterations; gibb_sample_i++){
			runGibbsSamplerForTreesAndSigsqOnce(gibb_sample_i);
			if (PrintOutEvery != null && gibb_sample_i % PrintOutEvery == 0){
				System.out.println("gibbs iter: " + gibb_sample_i + "/" + num_gibbs_total_iterations);
			}
		}
	}
	
	protected void InitiateGibbsChain() {
		
		//assign the first batch of trees by drawing from the prior and add it to the master list
		ArrayList<CGMTreeNode> initial_trees = createInitialTreesByDrawingFromThePrior();
		gibbs_samples_of_cgm_trees.add(0, initial_trees);	
		//now assign the first sigsq using the prior and add it to the master list
		double initial_sigsq = sampleInitialSigsqByDrawingFromThePrior();
//		System.out.println("initial_sigsq: " + initial_sigsq);
		gibbs_samples_of_sigsq.add(0, initial_sigsq);
		

		if (WRITE_DETAILED_DEBUG_FILES){
			//debug initial sigmas
			double[] initial_sigma_simus = new double[1000];
			for (int i = 0; i < 1000; i++){
				initial_sigma_simus[i] = sampleInitialSigsqByDrawingFromThePrior() * (TRANSFORM_Y ? y_range_sq : 1);
			}
//			System.out.print("\n\n\n");
			sigsqs_draws.println(0 + "," + hyper_nu + "," + hyper_lambda + "," + 0 + "," + 0 + "," + (initial_sigsq * (TRANSFORM_Y ? y_range_sq : 1)) + "," + y_range_sq + "," + IOTools.StringJoin(initial_sigma_simus, ","));		
		}
		
		//the structure is there (ie the T's) but the M's are not, so do that now:
		for (CGMTreeNode tree : initial_trees){
			assignLeafValsUsingPosteriorMeanAndCurrentSigsq(tree, initial_sigsq); //again, incorrect for the same reason as the first line of the function
		}		

		//DEBUG
		if (TREE_ILLUST){
			TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(0);
			for (CGMTreeNode tree : initial_trees){
				tree_array_illustration.AddTree(tree);
				tree_array_illustration.addLikelihood(0);			
			}
			tree_array_illustration.CreateIllustrationAndSaveImage();
		}
		
		if (WRITE_DETAILED_DEBUG_FILES){
			for (int t = 0; t < m; t++){
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
				sum_ys_without_jth_tree[i] += other_trees.get(t).Evaluate(X_y.get(i)); //first tree for now
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
	
	protected CGMTreeNode SampleTreeByCalculatingRemainingsAndDrawingFromTreeDist(final int prev_sample_num, final int t, TreeArrayIllustration tree_array_illustration) {
		final CGMTreeNode tree = gibbs_samples_of_cgm_trees.get(prev_sample_num).get(t);
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
		
		ArrayList<CGMTreeNode> other_trees = new ArrayList<CGMTreeNode>(m - 1);
		for (int j = 0; j < t; j++){
			other_trees.add(gibbs_samples_of_cgm_trees.get(prev_sample_num + 1).get(j));
		}
		for (int j = t + 1; j < m; j++){
			other_trees.add(gibbs_samples_of_cgm_trees.get(prev_sample_num).get(j));
		}		
		
		final double[] R_j = getResidualsBySubtractingTrees(other_trees);
		
//		System.out.println("SampleTreeByDrawingFromTreeDist rs = " + IOTools.StringJoin(R_j, ","));
		if (WRITE_DETAILED_DEBUG_FILES){
	//		new Thread(){
	//			public void run(){					
					remainings.println(prev_sample_num + "," + t + "," + tree.stringID() + "," + IOTools.StringJoin(R_j, ","));			
	//			}
	//		}.start();
		}
		
		//now, (important!) set the R_j's as this tree's data.
		tree.updateWithNewResponsesAndPropagate(X_y, R_j, p);
		
		//sample from T_j | R_j, \sigma
		//now we will run one M-H step on this tree with the y as the R_j
		//we first have to initialize the posterior builder in order to iterate
		CGMBARTPosteriorBuilder posterior_builder = new CGMBARTPosteriorBuilder(tree_prior_builder);
		//we have to set the CGM98 hyperparameters as well as the hyperparameter sigsq_mu
		posterior_builder.setHyperparameters(hyper_mu_mu, 1 / 3.0, hyper_nu, hyper_lambda, hyper_sigsq_mu);
		//we also need to set the current value of sigsq since we're conditioning on it
		posterior_builder.setCurrentSigsqValue(gibbs_samples_of_sigsq.get(prev_sample_num));
		//we name this iteration for better debugging
//			String iteration_name = "tree_" + CGMPosteriorBuilder.LeadingZeroes(t + 1, 4) + "_iter_" + CGMPosteriorBuilder.LeadingZeroes(i + 1, 6);
		//now we iterate one step
//		System.out.println("posterior_builder.calculateLnProbYGivenTree FIRST");
//		posterior_builder.calculateLnProbYGivenTree(tree);
		CGMTreeNode tree_star = posterior_builder.iterateMHPosteriorTreeSpaceSearch(tree, false);
		
		//DEBUG
//		System.err.println("tree star: " + tree_star.stringID());
		double lik = tree_star.log_prop_lik;
		tree_liks.print(lik + "," + tree_star.stringID() + ",");
		tree_array_illustration.addLikelihood(lik);
		all_tree_liks[t][prev_sample_num + 1] = lik;
		
		return tree_star;
	}	
	
	private ArrayList<CGMTreeNode> createInitialTreesByDrawingFromThePrior() {
//		System.out.println("CGMBART CreateInitialTrees numtrees: " + m);
		//initialize array for the first time
		ArrayList<CGMTreeNode> cgm_trees = new ArrayList<CGMTreeNode>(m);
		//now we're going to build each tree based on the prior given in section 2 of the paper
		//first thing is first, we create the tree structures using priors for p(T_1), p(T_2), .., p(T_m)
		
		for (int i = 0; i < m; i++){
//			System.out.println("CGMBART create prior on tree: " + (i + 1));
			CGMTreeNode tree = tree_prior_builder.buildTreeStructureBasedOnPrior();
			tree.initLogPropLik();
//			modifyTreeForDebugging(tree);
			tree.updateWithNewResponsesAndPropagate(X_y, y_trans, p);
			cgm_trees.add(tree);
		}
		return cgm_trees;
	}

//	private void modifyTreeForDebugging(CGMTreeNode tree) {
		//let's ensure the correct root node
//		tree.splitAttributeM = 0;
//		tree.splitValue = (double)30;
//		tree.isLeaf = false;
//	}	
}
