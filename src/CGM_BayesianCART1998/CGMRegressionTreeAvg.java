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
import java.util.Arrays;

import CGM_Statistics.*;

import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentView.JProgressBarAndLabel;

/**
 * This class takes trees from the Bayesian posterior and averages
 * 
 * @author Adam Kapelner
 *
 */
public class CGMRegressionTreeAvg extends CGMCART {
	private static final long serialVersionUID = -2492028493327218035L;

	private static final int DEFAULT_NUM_SEPARATE_CHAINS = 500;
	private static final int DEFAULT_MH_ITERATIONS_PER_CHAIN = 500;	
	private static final int DEFAULT_BURN_IN_PER_CHAIN = 200;
	private static final double DEFAULT_THIN_RATE_PER_CHAIN = 0.1;
	
	/** the root of this tree, overriden because we use a spruced-up node type */
	protected CGMTreeNode root;	
	/** the class that will create the prior tree */
	protected CGMTreePriorBuilder tree_prior_builder; 
	/** the class that will create the posterior tree */
	protected CGMPosteriorBuilder tree_posterior_builder;


	/** how many times should the algorithm be restarted? */
	private int num_separate_chains;
	/** for each time algorithm is started, we let it run this many MH iterations */
	private int num_iterations_per_chain;
	private int num_burn_in_per_chain;
	private double thin_rate_per_chain;
	/** stores all the trees for each iteration in each chain */
//	private ArrayList<ArrayList<CGMTreeNode>> all_trees_by_chain;
	/** stores all the trees for each iteration in each chain */
	private ArrayList<CGMTreeNode> final_candidate_trees;

	private boolean stop_building;


	public CGMRegressionTreeAvg(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
		final_candidate_trees = new ArrayList<CGMTreeNode>();
	}
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		tree_prior_builder = new CGMBayesianCARTPriorBuilder(X_y, p);
		num_separate_chains = DEFAULT_NUM_SEPARATE_CHAINS;
		num_iterations_per_chain = DEFAULT_MH_ITERATIONS_PER_CHAIN;	
		num_burn_in_per_chain = DEFAULT_BURN_IN_PER_CHAIN;
		thin_rate_per_chain = DEFAULT_THIN_RATE_PER_CHAIN;
		
		createPosteriorBuilder();		
	}	
	
	@Override
	protected void createPosteriorBuilder() {
		tree_posterior_builder = new CGMRegressionMeanShiftPosteriorBuilder(tree_prior_builder, y);	
	}

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
		//basically take average of the final trees
		return StatToolbox.sample_average(getTreeGuessesSorted(record, false));
	}	

	
	private double[] getTreeGuessesSorted(double[] record, boolean sort) {
		double[] yhats = new double[final_candidate_trees.size()];
		for (int i = 0; i < final_candidate_trees.size(); i++){
			yhats[i] = final_candidate_trees.get(i).Evaluate(record);
		}
		if (sort){
			Arrays.sort(yhats);
		}
		return yhats;
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
		runAllChainsAndBurnAndThinAndFlush();
//		dumpDataToFile();
	}

	private void runAllChainsAndBurnAndThinAndFlush() {
		for (int c = 1; c <= num_separate_chains; c++){
			ArrayList<CGMTreeNode> chain = new ArrayList<CGMTreeNode>(num_iterations_per_chain + 1);
			CGMTreeNode prior_tree = tree_prior_builder.buildTreeStructureBasedOnPrior();
			chain.add(prior_tree);
			for (int i = 0; i < num_iterations_per_chain; i++){
				chain.add(tree_posterior_builder.iterateMHPosteriorTreeSpaceSearch(chain.get(i), false));
				if (stop_building){
					break;
				}
			}
			System.out.println("done building chain num " + c);
			//now burn
			ArrayList<CGMTreeNode> temp_gibbs_samples_of_cgm_trees = new ArrayList<CGMTreeNode>(num_iterations_per_chain - num_burn_in_per_chain);
			for (int i = num_burn_in_per_chain; i < num_iterations_per_chain; i++){
				temp_gibbs_samples_of_cgm_trees.add(chain.get(i));
			}
			//now thin
			int thin_rate = (int)Math.round(temp_gibbs_samples_of_cgm_trees.size() * thin_rate_per_chain);
			for (int i = 0; i < temp_gibbs_samples_of_cgm_trees.size(); i++){
				if (i % thin_rate == 0){
					CGMTreeNode tree = temp_gibbs_samples_of_cgm_trees.get(i);
					CGMShared.assignLeaves(tree, this);
					final_candidate_trees.add(tree);
				}
			}
			System.out.println("done burning and thinning chain num " + c);
			//now assign leaves and clean up
			for (CGMTreeNode tree : final_candidate_trees){				
				tree.flushNodeData();
			}
			System.out.println("done assigning and cleaning up chain num " + c);
		}
	}

	@Override
	public void StopBuilding() {
		stop_building = true;
	}
	
	public void FlushData(){} //already taken care of during build

	@Override
	public void assignLeaf(CGMTreeNode node) {
		//for now just take mean of the y's... I don't think we can do anything else since all the values are margined out
		double[] y_is = node.get_ys_in_data();
		node.y_prediction = StatToolbox.sample_average(y_is);
	}
	
	private double[] getPostPredictiveIntervalForPrediction(double[] record, double coverage){
		//get all gibbs samples sorted
		double[] y_gibbs_samples_sorted = getTreeGuessesSorted(record, true);
		
		//calculate index of the CI_a and CI_b
		int n_bottom = (int)Math.round((1 - coverage) / 2 * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
		int n_top = (int)Math.round(((1 - coverage) / 2 + coverage) * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
		
		double[] conf_interval = {y_gibbs_samples_sorted[n_bottom], y_gibbs_samples_sorted[n_top]};
		System.out.println("getPostPredictiveIntervalForPrediction l=" + y_gibbs_samples_sorted.length + " n_a=" + n_bottom + " n_b=" + n_top + "  [" + conf_interval[0] + ", " + conf_interval[1] + "]");
		return conf_interval;
	}
	
	private double[] get95PctPostPredictiveIntervalForPrediction(double[] record){
		return getPostPredictiveIntervalForPrediction(record, 0.95);
	}	
	
	public void writeEvaluationDiagnostics() {
		output.print("y,yhat,a,b,inside");
		output.print("\n");
		for (int i=0; i<n; i++){
			double[] record = X_y.get(i);
			double y = getResponseFromRecord(record);
			double yhat = Evaluate(record);
			double[] ppi = get95PctPostPredictiveIntervalForPrediction(record);
			int inside = (y >= ppi[0] && y <= ppi[1]) ? 1 : 0;
			output.println(y + "," + yhat + "," + ppi[0] + "," + ppi[1] + "," + inside);
		}		
		output.close();
	}	
}
