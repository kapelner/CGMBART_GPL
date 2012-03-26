package CGM_BART;

import java.util.ArrayList;

import CGM_Statistics.CGMTreeNode;
import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentView.JProgressBarAndLabel;

public class CGMBART_FixedTree extends CGMBART2010 {
	private static final long serialVersionUID = 3460935328647793073L;
	private CGMBARTPosteriorBuilder posterior_builder;
	
	public CGMBART_FixedTree(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
		System.out.println("CGMBART_FixedTree init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
	}
	
	public void setNumTrees(int m){
		this.m = 1; //in this DEBUG model, there's only one tree, so make sure it can't be anything else
	}
	
	protected void InitiateGibbsChain() {
		
		posterior_builder = new CGMBARTPosteriorBuilder(tree_prior_builder);
		//we have to set the CGM98 hyperparameters as well as the hyperparameter sigsq_mu
		posterior_builder.setHyperparameters(hyper_mu_mu, hyper_nu, hyper_lambda, hyper_sigsq_mu);		
		
		//assign the first batch of trees by drawing from the prior and add it to the master list
		ArrayList<CGMTreeNode> initial_trees = new ArrayList<CGMTreeNode>(1);
		initial_trees.add(CreateTheSimpleTreeModel());
		gibbs_samples_of_cgm_trees.add(0, initial_trees);	
		//now assign the first sigsq using the prior and add it to the master list
		double initial_sigsq = sampleInitialSigsqByDrawingFromThePrior();
//		System.out.println("initial_sigsq: " + initial_sigsq);
		gibbs_samples_of_sigsq.add(0, initial_sigsq);
		
	}
	
	protected void runGibbsSamplerForTreesAndSigsqOnce(final int sample_num) {		
		
		tree_liks.print(sample_num + ",");
		
		//create the fixed tree struct
		final ArrayList<CGMTreeNode> cgm_trees = new ArrayList<CGMTreeNode>(1);
		final CGMTreeNode copy_of_old_tree = gibbs_samples_of_cgm_trees.get(sample_num - 1).get(0).clone();
		copy_of_old_tree.updateWithNewResponsesAndPropagate(X_y, y_trans, p); //no need for new y vector (which is usually the residuals from other trees)
		cgm_trees.add(copy_of_old_tree);
		gibbs_samples_of_cgm_trees.add(sample_num, cgm_trees);
		//calculate likelihood which is not a function of M
		double lik = posterior_builder.calculateLnProbYGivenTree(copy_of_old_tree);
		System.out.println("Running BART Gibbs sampler fixed tree, iteration " + sample_num + " lik = " + lik);
		tree_liks.print(lik + "," + copy_of_old_tree.stringID() + "\n");
		all_tree_liks[0][sample_num] = lik;
		
		//assign leaf values
		double previous_sigsq = gibbs_samples_of_sigsq.get(sample_num - 1);
		assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(copy_of_old_tree, previous_sigsq);

		//after there are leaf values, we can illustrate them if we like
		if (TREE_ILLUST){
			final TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(sample_num);			
			tree_array_illustration.AddTree(copy_of_old_tree);
			tree_array_illustration.addLikelihood(lik);
			tree_array_illustration.CreateIllustrationAndSaveImage();
		}
		
		//okay now we have to sample a posterior sigma given all the trees and add that to the gibbs pantheon
		double current_sigsq = samplePosteriorSigsq(sample_num);
//		System.out.println("current_sigsq = " + current_sigsq);
		
		gibbs_samples_of_sigsq.add(sample_num, current_sigsq);
		posterior_builder.setCurrentSigsqValue(current_sigsq);
		System.out.println("setCurrentSigsqValue transformed_sigsq = " + current_sigsq * y_range_sq + " sigsq_mu = " + hyper_sigsq_mu);
		
		sigsqs.println(sample_num + "," + gibbs_samples_of_sigsq.get(sample_num) * y_range_sq);	
		
		for (CGMTreeNode tree_it : cgm_trees){
			tree_it.flushNodeData();	
		}		

		//now close and open all debug
		if (Math.random() < 0.0333){
			CloseDebugFiles();
			OpenDebugFiles();
		}
	
	}		
	

}
