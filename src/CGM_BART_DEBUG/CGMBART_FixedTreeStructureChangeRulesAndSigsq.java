package CGM_BART_DEBUG;

import java.util.ArrayList;

import CGM_BART.*;

public class CGMBART_FixedTreeStructureChangeRulesAndSigsq extends CGMBART_09_eval {
	private static final long serialVersionUID = 3460935328647793073L;
	
	public CGMBART_FixedTreeStructureChangeRulesAndSigsq() {
		super();
		System.out.println("CGMBART_Alt init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
//		printTreeIllustations();
	}
	
	//start the tree with no information
	protected void InitiatizeTrees() {
		ArrayList<CGMBARTTreeNode> cgm_trees = new ArrayList<CGMBARTTreeNode>(num_trees);
		CGMBARTTreeNode tree = CGMBART_FixedTree.CreateTheSimpleTreeModel(this);
		tree.splitAttributeM = 0;
		tree.splitValue = 0.0;
		tree.left.splitAttributeM = 0;
		tree.left.splitValue = 0.0;
		tree.right.splitAttributeM = 0;
		tree.right.splitValue = 0.0;
		cgm_trees.add(tree);	
		gibbs_samples_of_cgm_trees.add(0, cgm_trees);		
	}
}
