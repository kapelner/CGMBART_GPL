package CGM_BART;

import java.util.ArrayList;
import java.io.Serializable;


public abstract class CGMBART_04_init extends CGMBART_03_debug implements Serializable {
	private static final long serialVersionUID = 8239599486635371714L;
	
	/** during debugging, we may want to fix sigsq */
	protected double fixed_sigsq;
	/** which gibbs sample are we on now? */
	protected int gibbs_sample_num;
	/** cached current sum resids_vec */
	protected double[] sum_resids_vec;	
	
	protected void SetupGibbsSampling(){
//		System.out.println("SetupGibbsSampling");
		InitGibbsSamplingData();	
		InitizializeSigsq();
		InitializeTrees();
		InitializeMus();		
		DebugInitialization();	
		//the zeroth gibbs sample is the initialization we just did; now we're onto the first in the chain
		gibbs_sample_num = 1;
		
		sum_resids_vec = new double[n];
	}
	
	protected void InitGibbsSamplingData(){
		//now initialize the gibbs sampler array for trees and error variances
		gibbs_samples_of_cgm_trees = new CGMBARTTreeNode[num_gibbs_total_iterations + 1][num_trees];
		gibbs_samples_of_cgm_trees_after_burn_in = new CGMBARTTreeNode[num_gibbs_total_iterations - num_gibbs_burn_in + 1][num_trees];
		gibbs_samples_of_sigsq = new double[num_gibbs_total_iterations + 1];	
		gibbs_samples_of_sigsq_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in];		
	}
	
	
	protected void InitializeTrees() {
		//create the array of trees for the zeroth gibbs sample
		CGMBARTTreeNode[] cgm_trees = new CGMBARTTreeNode[num_trees];		
		for (int i = 0; i < num_trees; i++){
			CGMBARTTreeNode stump = new CGMBARTTreeNode(this);
			stump.setStumpData(X_y, y_trans, p);
			cgm_trees[i] = stump;
		}	
		gibbs_samples_of_cgm_trees[0] = cgm_trees;	
	}

	protected static final double INITIAL_PRED = 0; //median, doesn't matter anyway
	protected void InitializeMus() {
//		System.out.println("InitializeMus");
		//we don't want to do this
//		for (CGMBARTTreeNode tree : gibbs_samples_of_cgm_trees.get(0)){
//			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(tree, gibbs_samples_of_sigsq.get(0));
//		}	
		for (CGMBARTTreeNode stump : gibbs_samples_of_cgm_trees[0]){
			stump.y_pred = INITIAL_PRED;
		}
	}
	
	protected static final double INITIAL_SIGSQ = Math.pow(0.5 / 3, 2); //median, doesn't matter anyway
	protected void InitizializeSigsq() {
//		System.out.println("InitizializeSigsq");
		gibbs_samples_of_sigsq[0] = INITIAL_SIGSQ;
//		gibbs_samples_of_sigsq.add(0, sampleInitialSigsqByDrawingFromThePrior());
	}
	
	protected double sampleInitialSigsqByDrawingFromThePrior() {
		//we're sampling from sigsq ~ InvGamma(nu / 2, nu * lambda / 2)
		//which is equivalent to sampling (1 / sigsq) ~ Gamma(nu / 2, 2 / (nu * lambda))
		return StatToolbox.sample_from_inv_gamma(hyper_nu / 2, 2 / (hyper_nu * hyper_lambda)); 
	}	

	protected abstract void assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqAndUpdateYhats(CGMBARTTreeNode node, double sigsq);
	
	
	public void setNumGibbsBurnIn(int num_gibbs_burn_in){
		this.num_gibbs_burn_in = num_gibbs_burn_in;
	}
	
	public void setNumGibbsTotalIterations(int num_gibbs_total_iterations){
		this.num_gibbs_total_iterations = num_gibbs_total_iterations;
	}
	
	public int numSamplesAfterBurningAndThinning(){
		return num_gibbs_total_iterations - num_gibbs_burn_in;
	}	
	
	public void setSigsq(double fixed_sigsq){
		this.fixed_sigsq = fixed_sigsq;
	}	
}
