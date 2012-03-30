package CGM_BART_DEBUG;

import java.util.ArrayList;

import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentView.JProgressBarAndLabel;
import CGM_BART.*;
import CGM_Statistics.CGMTreeNode;

public class CGMBART_Alt extends CGMBART {
	private static final long serialVersionUID = 3460935328647793073L;
	
	public CGMBART_Alt(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
		System.out.println("CGMBART_Alt init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
//		printTreeIllustations();
	}	
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		//this posterior builder will be shared throughout the entire process
		posterior_builder = new CGMBARTPosteriorBuilder_Alt(tree_prior_builder);
		//we have to set the CGM98 hyperparameters as well as the hyperparameter sigsq_mu
		posterior_builder.setHyperparameters(hyper_mu_mu, hyper_sigsq_mu);
	}

	//only the simple tree
	protected void InitiatizeTrees() {
		ArrayList<CGMTreeNode> cgm_trees = new ArrayList<CGMTreeNode>(num_trees);
		CGMTreeNode tree = CreateTheSimpleTreeModel();
		tree.splitAttributeM = 0;
		tree.splitValue = 0.0;
		tree.left.splitAttributeM = 0;
		tree.left.splitValue = 0.0;
		tree.right.splitAttributeM = 0;
		tree.right.splitValue = 0.0;
		cgm_trees.add(tree);	
		gibbs_samples_of_cgm_trees.add(0, cgm_trees);		
	}
//
//	private CGMTreeNode CreateOneBranchTree() {
//		CGMTreeNode root = new CGMTreeNode(null, null, this);
//		CGMTreeNode left = new CGMTreeNode(null, null, this);
//		CGMTreeNode right = new CGMTreeNode(null, null, this);
//
//		root.isLeaf = false;
//		root.splitAttributeM = 0;
//		root.splitValue = 0.0;
//		root.left = left;
//		root.right = right;	
//
//		left.parent = root;
//		left.isLeaf = true;
//
//		right.parent = root;
//		right.isLeaf = true;
//
//		//make sure there's data in there
//		root.updateWithNewResponsesAndPropagate(X_y, y_trans, p);
//		return root;
//	}
}
