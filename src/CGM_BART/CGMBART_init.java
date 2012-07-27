package CGM_BART;

import java.util.ArrayList;
import java.io.Serializable;

import CGM_Statistics.StatToolbox;

public abstract class CGMBART_init extends CGMBART_debug implements Serializable {
	private static final long serialVersionUID = 8239599486635371714L;

	protected int num_gibbs_burn_in;
	protected int num_gibbs_total_iterations;	
	/** during debugging, we may want to fix sigsq */
	protected double fixed_sigsq;
	/** which gibbs sample are we on now? */
	protected int gibb_sample_num;		
	
	public CGMBART_init() {
		super();
		num_gibbs_burn_in = DEFAULT_NUM_GIBBS_BURN_IN;
		num_gibbs_total_iterations = DEFAULT_NUM_GIBBS_TOTAL_ITERATIONS;
	}
	
	protected void SetupGibbsSampling(){
		InitGibbsSamplingData();	
		InitizializeSigsq();
		InitiatizeTrees();
		InitializeMus();		
		DebugInitialization();	
		gibb_sample_num = 1;
	}
	
	protected void InitGibbsSamplingData(){
		all_tree_liks = new double[num_trees][num_gibbs_total_iterations + 1];

		//now initialize the gibbs sampler array for trees and error variances
		gibbs_samples_of_cgm_trees = new ArrayList<ArrayList<CGMBARTTreeNode>>(num_gibbs_total_iterations);
		gibbs_samples_of_cgm_trees_after_burn_in = new ArrayList<ArrayList<CGMBARTTreeNode>>(num_gibbs_total_iterations - num_gibbs_burn_in);
		gibbs_samples_of_sigsq = new ArrayList<Double>(num_gibbs_total_iterations);	
		gibbs_samples_of_sigsq_after_burn_in = new ArrayList<Double>(num_gibbs_total_iterations - num_gibbs_burn_in);		
	}
	
	protected void InitiatizeTrees() {
		//create the array of trees for the zeroth gibbs sample
		ArrayList<CGMBARTTreeNode> cgm_trees = new ArrayList<CGMBARTTreeNode>(num_trees);		
		for (int i = 0; i < num_trees; i++){
//			System.out.println("CGMBART create prior on tree: " + (i + 1));
			CGMBARTTreeNode stump = new CGMBARTTreeNode(null, X_y, this);
			stump.y_prediction = 0.0; //default
//			modifyTreeForDebugging(tree);
			stump.updateWithNewResponsesAndPropagate(X_y, y_trans, p);
			cgm_trees.add(stump);
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
