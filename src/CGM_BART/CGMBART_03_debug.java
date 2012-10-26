package CGM_BART;

import java.io.Serializable;
import java.util.ArrayList;


public abstract class CGMBART_03_debug extends CGMBART_02_hyperparams implements Serializable {
	private static final long serialVersionUID = -5808113783423229776L;
	
	protected static boolean TREE_ILLUST = false;
	
	public static final String DEBUG_DIR = "debug_output";

	static {		
		TreeIllustration.DeletePreviousTreeIllustrations();
	}
	
	protected void DebugInitialization() {
		ArrayList<CGMBARTTreeNode> initial_trees = gibbs_samples_of_cgm_trees.get(0);
			
		if (TREE_ILLUST){
			TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(0, unique_name);
			for (CGMBARTTreeNode tree : initial_trees){
				tree_array_illustration.AddTree(tree);
				tree_array_illustration.addLikelihood(0);			
			}
			tree_array_illustration.CreateIllustrationAndSaveImage();
		}
	}
	
	public void printTreeIllustations(){
		TREE_ILLUST = true;
	}		

	protected void DebugSample(int gibbs_sample_num, TreeArrayIllustration tree_array_illustration) {
		if (TREE_ILLUST && gibbs_sample_num > 3900 && gibbs_sample_num < 4000 && num_trees == 1){ //
			tree_array_illustration.CreateIllustrationAndSaveImage();
		}
	}
	
	public double[] getGibbsSamplesSigsqs(){
		double[] sigsqs_to_export = new double[gibbs_samples_of_sigsq.size()];
		for (int n_g = 0; n_g < gibbs_samples_of_sigsq.size(); n_g++){			
			sigsqs_to_export[n_g] = gibbs_samples_of_sigsq.get(n_g) * y_range_sq;	//Var[y^t] = Var[y / R_y] = 1/R_y^2 Var[y]		
		}
		return sigsqs_to_export;
	}	
	
	private int maximalTreeGeneration(){
		int max_gen = Integer.MIN_VALUE;
		for (ArrayList<CGMBARTTreeNode> cgm_trees : gibbs_samples_of_cgm_trees){
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
	
	public double[] getLikForTree(int t){
		return all_tree_liks[t];
	}
	
	public int[] getNumNodesForTreesInGibbsSamp(int n_g){
		ArrayList<CGMBARTTreeNode> trees = gibbs_samples_of_cgm_trees.get(n_g);
		int[] num_nodes_by_tree = new int[trees.size()];
		for (int t = 0; t < trees.size(); t++){
			num_nodes_by_tree[t] = trees.get(t).numLeaves();
		}
		return num_nodes_by_tree;
	}	
	
	public int[] getDepthsForTreesInGibbsSamp(int n_g){
		ArrayList<CGMBARTTreeNode> trees = gibbs_samples_of_cgm_trees.get(n_g);
		int[] depth_by_tree = new int[trees.size()];
		for (int t = 0; t < trees.size(); t++){
			depth_by_tree[t] = trees.get(t).deepestNode();
		}
		return depth_by_tree;
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
