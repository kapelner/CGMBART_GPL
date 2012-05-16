package CGM_BART_DEBUG;

import CGM_BART.*;
import java.util.ArrayList;

import CGM_Statistics.CGMTreeNode;
import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentView.JProgressBarAndLabel;

public class CGMBART_FixedTreeAndMus extends CGMBART {
	private static final long serialVersionUID = -331480664944699926L;
	
	public CGMBART_FixedTreeAndMus(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
		System.out.println("CGMBART_FixedTreeAndLeaves init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
	}
	
	//only the simple tree
	protected void InitiatizeTrees() {
		ArrayList<CGMTreeNode> cgm_trees = new ArrayList<CGMTreeNode>(num_trees);
		cgm_trees.add(CreateTheSimpleTreeModel());	
		gibbs_samples_of_cgm_trees.add(0, cgm_trees);		
	}

	protected void SampleTree(int sample_num, int t, ArrayList<CGMTreeNode> cgm_trees, TreeArrayIllustration tree_array_illustration) {
		CGMTreeNode tree = CreateTheSimpleTreeModel();
		tree.updateWithNewResponsesAndPropagate(X_y, y_trans, p); //no need for new y vector (which is usually the residuals from other trees)
		cgm_trees.add(tree);
		gibbs_samples_of_cgm_trees.set(sample_num, cgm_trees);
		
		//The rest is all debug		
		double lik = posterior_builder.calculateLnProbYGivenTree(tree);
		tree_array_illustration.AddTree(tree);
		tree_array_illustration.addLikelihood(lik);
		System.out.println("Running BART Gibbs sampler fixed tree and mu's, iteration " + sample_num + " lik = " + lik);
		tree_liks.print(lik + "," + tree.stringID() + ",");
		all_tree_liks[0][sample_num] = lik;		
	}	

	protected void SampleMus(int sample_num, int t){
		//do nothing -- the structure is in the sample of the tree
	}
	
}
