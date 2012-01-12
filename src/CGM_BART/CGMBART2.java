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

@SuppressWarnings("serial")
public abstract class CGMBART2 extends CGMBART {

	public CGMBART2(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
	}

	protected void CoreBuild(){
		
		//first create a bunch of dumb trees as the "0th" Gibbs sample just to start the MCMC off...
		ArrayList<CGMTreeNode> initial_trees = new ArrayList<CGMTreeNode>(m);
		for (int j = 0; j < m; j++){
			initial_trees.add(createDumbDumbTree());
		}
		gibbs_samples_of_cgm_trees.add(0, initial_trees);

		//second step... draw sigma given the "prior" trees
		gibbs_samples_of_sigsq.add(0, samplePosteriorSigsq(0));

		//now we're off to the races, gibbs sampling away. We begin at sample **1** since we've already built the zeroth sample from the prior
		for (gibb_sample_i = 1; gibb_sample_i <= num_gibbs_total_iterations; gibb_sample_i++){
//			dumpDataToFile("_before_samp_" + sample_num); ////DATA IS GETTING EDITED!!!
			runGibbsSamplerForTreesAndSigsqOnce(gibb_sample_i);
		}		
	}
	
	private static final int NUM_SAMPLES_FOR_MH_TO_GET_NEW_TREE = 10;
	protected CGMTreeNode SampleTreeByCalculatingRemainingsAndDrawingFromTreeDist(final int prev_sample_num, final int t, TreeArrayIllustration tree_array_illustration) {
		int current_sample_num = prev_sample_num + 1;
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
			other_trees.add(gibbs_samples_of_cgm_trees.get(current_sample_num).get(j));
		}
		for (int j = t + 1; j < m; j++){
			other_trees.add(gibbs_samples_of_cgm_trees.get(prev_sample_num).get(j));
		}		
		
		final double[] R_j = getResidualsBySubtractingTrees(other_trees);
		
		/////NOT ACTUALLY CORRECT since it's using the full data (without subtracting off the
		/////effects of other trees) BUT PROBABLY GOOD ENOUGH since we're going to be running lots of samples
		final CGMTreeNode tree = tree_prior_builder.buildTreeStructureBasedOnPrior();
		
		//now, (important!) set the R_j's as this tree's data.
		tree.updateWithNewResponsesAndPropagate(X_y, R_j, p);
		
		
//		System.out.println("SampleTreeByDrawingFromTreeDist rs = " + IOTools.StringJoin(R_j, ","));
		if (WRITE_DETAILED_DEBUG_FILES){
			new Thread(){
				public void run(){
					remainings.println((prev_sample_num + 1) + "," + t + "," + tree.stringID() + "," + IOTools.StringJoin(R_j, ","));			
				}
			}.start();
		}
		

		
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
		CGMTreeNode tree_star = posterior_builder.convergePosteriorAndFindMostLikelyTree(tree, NUM_SAMPLES_FOR_MH_TO_GET_NEW_TREE, 0);
		
		//DEBUG
//		System.err.println("tree star: " + tree_star.stringID());
		double lik = posterior_builder.calculateLnProbYGivenTree(tree_star);
		tree_liks.print(lik + "," + tree_star.stringID() + ",");
		tree_array_illustration.addLikelihood(lik);
		
		return tree_star;		
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
//		System.out.println("getResidualsForAllTreesExcept " + (j+1) + "th tree:  " +  new DoubleMatrix(Rjs).transpose().toString(2));
		return Rjs;
	}

	private CGMTreeNode createDumbDumbTree(){
		CGMTreeNode dumbdumb = new CGMTreeNode(null, X_y);
		dumbdumb.y_prediction = 0.0; //since the observations are centered, have each tree predict zero
		return dumbdumb;
	}
}
