package CGM_BART_DEBUG;

import CGM_BART.*;

public class CGMBART_FixedTreeInitially extends CGMBART_09_eval {
	private static final long serialVersionUID = 3460935328647793073L;
	
	public CGMBART_FixedTreeInitially() {
		super();
		System.out.println("CGMBART_FixedTreeInitially init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
//		printTreeIllustations();
	}

	//only the simple tree as the initial seed
	protected void InitializeTrees() {
		CGMBARTTreeNode[] cgm_trees = new CGMBARTTreeNode[num_trees];
		cgm_trees[0] = CGMBART_FixedTree.CreateTheSimpleTreeModel(this);	
		gibbs_samples_of_cgm_trees[0] = cgm_trees;		
	}	

}
