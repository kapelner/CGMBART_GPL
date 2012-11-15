package CGM_BART_DEBUG;

import CGM_BART.*;

public class CGMBART_FixedSigsqAndTreeStructureJustChanges extends CGMBART_09_eval {
	private static final long serialVersionUID = 3460935328647793073L;
	
	public CGMBART_FixedSigsqAndTreeStructureJustChanges() {
		super();
		System.out.println("CGMBART_Alt init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
//		printTreeIllustations();
	}
	

	//fix it once for good
	protected void InitizializeSigsq() {	
		fixed_sigsq = 1 / y_range_sq;
		gibbs_samples_of_sigsq[0] = fixed_sigsq;
	}	
	
	//keep sigsq fixed
	protected void SampleSigsq(int sample_num) {
		gibbs_samples_of_sigsq[sample_num] = fixed_sigsq; //fix it forever
	}	

	//start the tree with no information
	protected void InitializeTrees() {
		CGMBARTTreeNode[] cgm_trees = new CGMBARTTreeNode[num_trees];
		CGMBARTTreeNode tree = CGMBART_FixedTree.CreateTheSimpleTreeModel(this);
		tree.splitAttributeM = 0;
		tree.splitValue = 0.0;
		tree.left.splitAttributeM = 0;
		tree.left.splitValue = 0.0;
		tree.right.splitAttributeM = 0;
		tree.right.splitValue = 0.0;
		cgm_trees[0] = tree;	
		gibbs_samples_of_cgm_trees[0] = cgm_trees;		
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
