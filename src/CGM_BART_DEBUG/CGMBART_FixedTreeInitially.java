package CGM_BART_DEBUG;

import CGM_BART.*;

import java.util.ArrayList;


public class CGMBART_FixedTreeInitially extends CGMBART_09_eval {
	private static final long serialVersionUID = 3460935328647793073L;
	
	public CGMBART_FixedTreeInitially() {
		super();
		System.out.println("CGMBART_FixedTreeInitially init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
//		printTreeIllustations();
	}

	//only the simple tree as the initial seed
	protected void InitiatizeTrees() {
		ArrayList<CGMBARTTreeNode> cgm_trees = new ArrayList<CGMBARTTreeNode>(num_trees);
		cgm_trees.add(CGMBART_FixedTree.CreateTheSimpleTreeModel(this));	
		gibbs_samples_of_cgm_trees.add(0, cgm_trees);		
	}	

}
