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

package CGM_BayesianCART1998;

import java.util.ArrayList;

import CGM_Statistics.*;
import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentView.JProgressBarAndLabel;

public abstract class CGMCART extends ClassificationAndRegressionTree implements LeafAssigner {
	private static final long serialVersionUID = 9148052856177317690L;
	

	/** the root of this tree, overriden because we use a spruced-up node type */
	protected CGMTreeNode root;	
	/** the class that will create the prior tree */
	protected CGMTreePriorBuilder tree_prior_builder; 
	/** the class that will create the posterior tree */
	protected CGMPosteriorBuilder tree_posterior_builder;


	/** how many times should the algorithm be restarted? */
	private int num_restarts;
	/** for each time algorithm is started, we let it run this many MH iterations */
	private int num_iterations;
	
	private static final int DEFAULT_NUM_RESTARTS_MH_ALGORITHM = 1;
	private static final int DEFAULT_MH_ITERATIONS_PER_SET = 10000;	

	public CGMCART(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);

	}
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		tree_prior_builder = new CGMBayesianCARTPriorBuilder(X_y, p);
		num_restarts = DEFAULT_NUM_RESTARTS_MH_ALGORITHM;
		num_iterations = DEFAULT_MH_ITERATIONS_PER_SET;	
		createPosteriorBuilder();
	}	
	
	protected abstract void createPosteriorBuilder();

	/**
	 * This will classify a new data record by using tree
	 * recursion and testing the relevant variable at each node.
	 * 
	 * This is probably the most-used function in all of <b>GemIdent</b>.
	 * It would make sense to inline this in assembly for optimal performance.
	 * 
	 * @param record 	the data record to be classified
	 * @return			the class the data record was classified into
	 */
	public double Evaluate(double[] record){ 
		return root.Evaluate(record);
	}	

	
	/**
	 * very simple: first build the prior tree then 
	 * take care of the prior updating via M-H algorithm,
	 * then assign classes to the leaves of the final tree,
	 * which will make it viable for evaluation, then flush the 
	 * data for cleanup
	 */
	@Override
	public void Build() {
		root = convergePosteriorMultipleTimesAndTakeMax();
//		dumpDataToFile();
		CGMShared.assignLeaves(root, this);
//		CGMTreeNode.DebugWholeTree(root);		
		FlushData();	
	}
	
	private CGMTreeNode convergePosteriorMultipleTimesAndTakeMax() {
		//run each block and save the cream of that block
		CGMTreeNode[] best_tree_by_restart = new CGMTreeNode[num_restarts];
		for (int num_restart = 0; num_restart < num_restarts; num_restart++){
			CGMTreeNode prior_tree = tree_prior_builder.buildTreeStructureBasedOnPrior();
//			System.out.println("convergePosteriorMultipleTimesAndTakeMax  node:" + prior_tree.stringID() + " leaf:" + prior_tree.isLeaf + " left: " + prior_tree.left + " right:" + prior_tree.right);
//			System.out.println("likelihood of " + i + "th prior tree: " + tree_posterior_builder.calculateLnProbYGivenTree(prior_tree));
			best_tree_by_restart[num_restart] = tree_posterior_builder.convergePosteriorAndFindMostLikelyTree(prior_tree, num_iterations, num_restart + 1);
		}
		tree_posterior_builder.close_debug_information();

		//now search over all restart blocks		
		System.out.println("Begin search over all " + num_restarts + " restarts");
		double highest_log_probability = -Double.MAX_VALUE;
		CGMTreeNode global_best_tree = null;
		for (int i = 0; i < num_restarts; i++){
			CGMTreeNode best_tree = best_tree_by_restart[i];
			double ln_prob_y_proposal = tree_posterior_builder.calculateLnProbYGivenTree(best_tree);
			System.out.println("evaluate iteration block " + (i + 1) + " best tree log-likelihood: " + ln_prob_y_proposal);
			if (ln_prob_y_proposal > highest_log_probability){	
				System.out.println("most likely tree log-likelihood: " + ln_prob_y_proposal);
				highest_log_probability = ln_prob_y_proposal;
				global_best_tree = best_tree.clone(true);
			}			
		}
		System.out.println("global best tree: " + highest_log_probability);		
		return global_best_tree;
	}

	@Override
	public void StopBuilding() {
		if (root != null){
			tree_posterior_builder.StopBuilding();
		}
	}
	
	public void FlushData(){
		root.flushNodeData();
	}
}
