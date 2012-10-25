package CGM_BART_DEBUG;

import CGM_BART.*;

import java.util.ArrayList;

public class CGMBART_FixedTree extends CGMBART_09_eval {
	private static final long serialVersionUID = 3460935328647793073L;
	
	public CGMBART_FixedTree() {
		super();
		System.out.println("CGMBART_FixedTree init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
		
	}

	//only the simple tree 
	protected void InitializeTrees() {
		ArrayList<CGMBARTTreeNode> cgm_trees = new ArrayList<CGMBARTTreeNode>(num_trees);
		cgm_trees.add(CreateTheSimpleTreeModel(this));	
		gibbs_samples_of_cgm_trees.add(0, cgm_trees);		
	}

	protected double[] SampleTree(int sample_num, int t, ArrayList<CGMBARTTreeNode> cgm_trees, TreeArrayIllustration tree_array_illustration) {
		CGMBARTTreeNode tree = CreateTheSimpleTreeModel(this);
		tree.updateWithNewResponsesRecursively(y_trans); //no need for new y vector (which is usually the residuals from other trees)
		cgm_trees.add(tree);
		gibbs_samples_of_cgm_trees.set(sample_num, cgm_trees);
		
		//The rest is all debug 
		double lik = 0;//posterior_builder.calculateLnProbYGivenTree(tree);
		tree_array_illustration.AddTree(tree);
		tree_array_illustration.addLikelihood(lik);
		System.out.println("Running BART Gibbs sampler fixed tree and mu's, iteration " + sample_num + " lik = " + lik);
//		tree_liks.print(lik + "," + tree.stringID() + ",");
		all_tree_liks[0][sample_num] = lik;	
		
		return y_trans;
	}	
	
	
	public static CGMBARTTreeNode CreateTheSimpleTreeModel(CGMBART_09_eval bart) {
		CGMBARTTreeNode root = new CGMBARTTreeNode(null, bart);
		CGMBARTTreeNode left = new CGMBARTTreeNode(null, bart);
		CGMBARTTreeNode leftleft = new CGMBARTTreeNode(null, bart);
		CGMBARTTreeNode leftright = new CGMBARTTreeNode(null, bart);
		CGMBARTTreeNode right = new CGMBARTTreeNode(null, bart);
		CGMBARTTreeNode rightleft = new CGMBARTTreeNode(null, bart);
		CGMBARTTreeNode rightright = new CGMBARTTreeNode(null, bart);

		root.isLeaf = false;
		root.splitAttributeM = 0;
		root.splitValue = 30.0;
		root.left = left;
		root.right = right;	

		left.parent = root;
		left.isLeaf = false;
		left.splitAttributeM = 2;
		left.splitValue = 10.0;	
		left.left = leftleft;
		left.right = leftright;

		leftleft.parent = left;
		leftleft.isLeaf = true;
		leftleft.y_pred = bart.transform_y(10);		

		leftright.parent = left;
		leftright.isLeaf = true;
		leftright.y_pred = bart.transform_y(30);

		right.parent = root;
		right.isLeaf = false;
		right.splitAttributeM = 1;
		right.splitValue = 80.0;	
		right.left = rightleft;
		right.right = rightright;

		rightleft.parent = right;
		rightleft.isLeaf = true;
		rightleft.y_pred = bart.transform_y(50);		

		rightright.parent = right;
		rightright.isLeaf = true;
		rightright.y_pred = bart.transform_y(70);

		//make sure there's data in there
		root.updateWithNewResponsesRecursively(bart.getYTrans());
		return root;
	}
}
