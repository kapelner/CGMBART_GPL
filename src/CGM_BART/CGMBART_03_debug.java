package CGM_BART;

import java.io.Serializable;

public abstract class CGMBART_03_debug extends CGMBART_02_hyperparams implements Serializable {
	private static final long serialVersionUID = -5808113783423229776L;
	
	protected boolean tree_illust = false;
	
	public static final String DEBUG_DIR = "debug_output";

	static {		
		TreeIllustration.DeletePreviousTreeIllustrations();
	}
	
	protected void DebugInitialization() {
		CGMBARTTreeNode[] initial_trees = gibbs_samples_of_cgm_trees[0];
			
		if (tree_illust){
			TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(0, unique_name);
			for (CGMBARTTreeNode tree : initial_trees){
				tree_array_illustration.AddTree(tree);
				tree_array_illustration.addLikelihood(0);			
			}
			tree_array_illustration.CreateIllustrationAndSaveImage();
		}
	}
	
	public void printTreeIllustations(){
		tree_illust = true;
	}		

	protected void DebugSample(int gibbs_sample_num, TreeArrayIllustration tree_array_illustration) {
		if (tree_illust){ //
			tree_array_illustration.CreateIllustrationAndSaveImage();
		}
	}
	
	public double[] getGibbsSamplesSigsqs(){
		double[] sigsqs_to_export = new double[gibbs_samples_of_sigsq.length];
		for (int n_g = 0; n_g < gibbs_samples_of_sigsq.length; n_g++){			
			sigsqs_to_export[n_g] = un_transform_sigsq(gibbs_samples_of_sigsq[n_g]);		
		}
		return sigsqs_to_export;
	}	
	
	private int maximalTreeGeneration(){
		int max_gen = Integer.MIN_VALUE;
		for (CGMBARTTreeNode[] cgm_trees : gibbs_samples_of_cgm_trees){
			if (cgm_trees != null){				
				for (CGMBARTTreeNode tree : cgm_trees){					
					int gen = tree.deepestNode();
					if (gen >= max_gen){
						max_gen = gen;
					}
				}
			}
		}
		return max_gen;
	}
	
	public int maximalNodeNumber(){
		int max_gen = maximalTreeGeneration();
		int node_num = 0;
		for (int g = 0; g <= max_gen; g++){
			node_num += (int)Math.pow(2, g);
		}
		return node_num;
	}	
	
	public int[][] getDepthsForTrees(int n_g_i, int n_g_f){
		int[][] all_depths = new int[n_g_f - n_g_i][num_trees];
		for (int g = n_g_i; g < n_g_f; g++){
			for (int t = 0; t < num_trees; t++){
				all_depths[g - n_g_i][t] = gibbs_samples_of_cgm_trees[g][t].deepestNode();
			}
		}
		return all_depths;
	}
	
	
	public int[][] getNumNodesAndLeavesForTrees(int n_g_i, int n_g_f){
		int[][] all_new_nodes = new int[n_g_f - n_g_i][num_trees];
		for (int g = n_g_i; g < n_g_f; g++){
			for (int t = 0; t < num_trees; t++){
				all_new_nodes[g - n_g_i][t] = gibbs_samples_of_cgm_trees[g][t].numNodesAndLeaves();
			}
		}
		return all_new_nodes;
	}	
	
//	public String getRootSplits(int n_g){
//		ArrayList<CGMBARTTreeNode> trees = gibbs_samples_of_cgm_trees.get(n_g);
//		ArrayList<String> root_splits = new ArrayList<String>(trees.size());
//		for (int t = 0; t < trees.size(); t++){
//			root_splits.add(trees.get(t).splitToString());
//		}
//		return Tools.StringJoin(root_splits, "   ||   ");		
//	}	

//	protected String treeIDsInCurrentSample(int sample_num){
//		ArrayList<CGMBARTTreeNode> trees = gibbs_samples_of_cgm_trees.get(sample_num);
//		ArrayList<String> treeIds = new ArrayList<String>(trees.size());
//		for (int t = 0; t < trees.size(); t++){
//			treeIds.add(trees.get(t).stringID());
//		}
//		return Tools.StringJoin(treeIds, ",");
//	}


}
