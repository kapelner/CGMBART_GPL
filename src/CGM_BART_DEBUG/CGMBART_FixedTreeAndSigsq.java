package CGM_BART_DEBUG;

import CGM_BART.*;

import java.util.ArrayList;


public class CGMBART_FixedTreeAndSigsq extends CGMBART_eval {
	private static final long serialVersionUID = -331480664944699926L;
	
	public CGMBART_FixedTreeAndSigsq() {
		super();
		System.out.println("CGMBART_FixedTreeAndSigsq init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree	
	}
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		//now calculate the fixed sigsq
		fixed_sigsq = 1 / y_range_sq;
		posterior_builder.setCurrentSigsqValue(fixed_sigsq);
	}
	
	protected void InitiatizeTrees() {
		ArrayList<CGMBARTTreeNode> cgm_trees = new ArrayList<CGMBARTTreeNode>(num_trees);
		cgm_trees.add(CGMBART_FixedTree.CreateTheSimpleTreeModel(this));	
		gibbs_samples_of_cgm_trees.add(0, cgm_trees);		
	}

	//fix it once for good
	protected void InitizializeSigsq() {		
		gibbs_samples_of_sigsq.add(0, fixed_sigsq);		
	}	
	
	protected void SampleSigsq(int sample_num) {
		gibbs_samples_of_sigsq.add(sample_num, fixed_sigsq); //fix it forever
	}

	protected void SampleTree(int sample_num, int t, ArrayList<CGMBARTTreeNode> cgm_trees, TreeArrayIllustration tree_array_illustration) {
		CGMBARTTreeNode tree = CGMBART_FixedTree.CreateTheSimpleTreeModel(this);
		tree.updateWithNewResponsesAndPropagate(X_y, y_trans, p); //no need for new y vector (which is usually the residuals from other trees)
		cgm_trees.add(tree);
		gibbs_samples_of_cgm_trees.set(sample_num, cgm_trees);
		
		//The rest is all debug
		double lik = 0; //posterior_builder.calculateLnProbYGivenTree(tree);
		tree_array_illustration.AddTree(tree);
		tree_array_illustration.addLikelihood(lik);
		System.out.println("Running BART Gibbs sampler fixed tree and sigsq, iteration " + sample_num + " lik = " + lik);
		tree_liks.print(lik + "," + tree.stringID() + ",");
		all_tree_liks[0][sample_num] = lik;		
	}

}
