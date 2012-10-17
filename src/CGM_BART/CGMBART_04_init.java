package CGM_BART;

import java.util.ArrayList;
import java.io.Serializable;


public abstract class CGMBART_04_init extends CGMBART_03_debug implements Serializable {
	private static final long serialVersionUID = 8239599486635371714L;

	protected int num_gibbs_burn_in;
	protected int num_gibbs_total_iterations;	
	/** during debugging, we may want to fix sigsq */
	protected double fixed_sigsq;
	/** which gibbs sample are we on now? */
	protected int gibb_sample_num;		
	
	public CGMBART_04_init() {		
		super();
		num_gibbs_burn_in = DEFAULT_NUM_GIBBS_BURN_IN;
		num_gibbs_total_iterations = DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS;
	}
	
	protected void SetupGibbsSampling(){
//		System.out.println("SetupGibbsSampling");
		InitGibbsSamplingData();	
		InitizializeSigsq();
		InitializeTrees();
		InitializeMus();		
		DebugInitialization();	
		//the zeroth gibbs sample is the initialization we just did; now we're onto the first in the chain
		gibb_sample_num = 1;
	}
	
	protected void InitGibbsSamplingData(){
//		System.out.println("InitGibbsSamplingData");
		all_tree_liks = new double[num_trees][num_gibbs_total_iterations + 1];

		//now initialize the gibbs sampler array for trees and error variances
		gibbs_samples_of_cgm_trees = new ArrayList<ArrayList<CGMBARTTreeNode>>(num_gibbs_total_iterations);
		gibbs_samples_of_cgm_trees_after_burn_in = new ArrayList<ArrayList<CGMBARTTreeNode>>(num_gibbs_total_iterations - num_gibbs_burn_in);
		gibbs_samples_of_sigsq = new ArrayList<Double>(num_gibbs_total_iterations);	
		gibbs_samples_of_sigsq_after_burn_in = new ArrayList<Double>(num_gibbs_total_iterations - num_gibbs_burn_in);		
	}
	
	
	protected void InitializeTrees() {
		//create the array of trees for the zeroth gibbs sample
		ArrayList<CGMBARTTreeNode> cgm_trees = new ArrayList<CGMBARTTreeNode>(num_trees);		
		for (int i = 0; i < num_trees; i++){
			CGMBARTTreeNode stump = new CGMBARTTreeNode(this);
			stump.setStumpData(X_y, y_trans, p);
			cgm_trees.add(stump);
		}	
		gibbs_samples_of_cgm_trees.add(cgm_trees);	
	}

	protected static final double INITIAL_PRED = 0; //median, doesn't matter anyway
	protected void InitializeMus() {
//		System.out.println("InitializeMus");
		//we don't want to do this
//		for (CGMBARTTreeNode tree : gibbs_samples_of_cgm_trees.get(0)){
//			assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(tree, gibbs_samples_of_sigsq.get(0));
//		}	
		for (CGMBARTTreeNode stump : gibbs_samples_of_cgm_trees.get(0)){
			stump.y_pred = INITIAL_PRED;
		}
	}
	
	protected static final double INITIAL_SIGSQ = Math.pow(0.5 / 3, 2); //median, doesn't matter anyway
	protected void InitizializeSigsq() {
//		System.out.println("InitizializeSigsq");
		gibbs_samples_of_sigsq.add(0, INITIAL_SIGSQ);
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
