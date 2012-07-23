package CGM_BART;

import java.util.ArrayList;
import java.io.Serializable;

import CGM_Statistics.StatToolbox;

public abstract class CGMBART_init extends CGMBART_debug implements Serializable {
	private static final long serialVersionUID = 8239599486635371714L;

	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);		
		//this posterior builder will be shared throughout the entire process
		posterior_builder = new CGMBARTPosteriorBuilder(this);
		//we have to set the CGM98 hyperparameters as well as the hyperparameter sigsq_mu
		posterior_builder.setHyperparameters(hyper_mu_mu, hyper_sigsq_mu);	
	}	

	protected void SetupGibbsSampling(){
		all_tree_liks = new double[num_trees][num_gibbs_total_iterations + 1];

		//now initialize the gibbs sampler array for trees and error variances
		gibbs_samples_of_cgm_trees = new ArrayList<ArrayList<CGMBARTTreeNode>>(num_gibbs_total_iterations);
		gibbs_samples_of_cgm_trees_after_burn_in = new ArrayList<ArrayList<CGMBARTTreeNode>>(num_gibbs_total_iterations - num_gibbs_burn_in);
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
		gibbs_samples_of_cgm_trees.add(cgm_trees);		
	}

	protected void InitializeMus() {
		for (CGMBARTTreeNode tree : gibbs_samples_of_cgm_trees.get(0)){
			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(tree, gibbs_samples_of_sigsq.get(0));
		}		
	}
	
	protected void InitizializeSigsq() {
		gibbs_samples_of_sigsq.add(0, sampleInitialSigsqByDrawingFromThePrior());		
	}
	
	protected double sampleInitialSigsqByDrawingFromThePrior() {
		//we're sampling from sigsq ~ InvGamma(nu / 2, nu * lambda / 2)
		//which is equivalent to sampling (1 / sigsq) ~ Gamma(nu / 2, 2 / (nu * lambda))
		return StatToolbox.sample_from_inv_gamma(hyper_nu / 2, 2 / (hyper_nu * hyper_lambda)); 
	}	

	protected abstract void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(CGMBARTTreeNode node, double sigsq);
}
