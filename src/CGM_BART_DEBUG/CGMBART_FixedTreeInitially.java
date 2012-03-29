package CGM_BART_DEBUG;

import CGM_BART.*;
import java.util.ArrayList;

import CGM_Statistics.CGMTreeNode;
import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentView.JProgressBarAndLabel;

public class CGMBART_FixedTreeInitially extends CGMBART {
	private static final long serialVersionUID = 3460935328647793073L;
	
	public CGMBART_FixedTreeInitially(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
		System.out.println("CGMBART_FixedTreeInitially init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
//		printTreeIllustations();
	}

	//only the simple tree as the initial seed
	protected void InitiatizeTrees() {
		ArrayList<CGMTreeNode> cgm_trees = new ArrayList<CGMTreeNode>(num_trees);
		cgm_trees.add(CreateTheSimpleTreeModel());	
		gibbs_samples_of_cgm_trees.add(0, cgm_trees);		
	}	

}
