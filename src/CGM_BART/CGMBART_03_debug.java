package CGM_BART;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;


public abstract class CGMBART_03_debug extends CGMBART_02_hyperparams implements Serializable {
	private static final long serialVersionUID = -5808113783423229776L;

	protected static PrintWriter y_and_y_trans;
	protected static PrintWriter sigsqs;
	protected static PrintWriter other_debug;
	protected static PrintWriter sigsqs_draws;
	protected static PrintWriter tree_liks;
	protected static PrintWriter remainings;
	protected static PrintWriter evaluations;
	public static PrintWriter mh_iterations_full_record;
	
	protected static boolean TREE_ILLUST = false;
	protected static final boolean WRITE_DETAILED_DEBUG_FILES = false;
	
	public static final String DEBUG_DIR = "debug_output";

	static {
		try {			
			output = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "output" + DEBUG_EXT)));
			other_debug = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "other_debug" + DEBUG_EXT)));
			y_and_y_trans = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "y_and_y_trans" + DEBUG_EXT)));
			sigsqs = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "sigsqs" + DEBUG_EXT)));
			sigsqs.println("sample_num,sigsq");
			sigsqs_draws = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "sigsqs_draws" + DEBUG_EXT)));
			double[] simu = new double[1000];
			for (int i = 1; i <= 1000; i++){
				simu[i-1] = i;
			}			
			sigsqs_draws.println("sample_num,nu,lambda,n,sse,realization,corr," + Tools.StringJoin(simu, ","));			
			tree_liks = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "tree_liks" + DEBUG_EXT)));
			evaluations = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "evaluations" + DEBUG_EXT)));
			remainings = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "remainings" + DEBUG_EXT)));
			tree_liks.print("sample_num,");
			for (int t = 0; t < DEFAULT_NUM_TREES; t++){
				tree_liks.print("t_" + t + "_lik,t_" + t + "_id,");
			}
			tree_liks.print("\n");
			mh_iterations_full_record = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "mh_iterations_full_record" + DEBUG_EXT)));
			mh_iterations_full_record.println(
					"step" + "," + 
					"node_to_change" + "," + 
					"loc" + "," +
					"a_i" + "," +
					"v_i" + "," +
					"a_*" + "," +	
					"v_*" + "," +
					"leaf_1_*" + "," + 
					"leaf_2_*" + "," + 
					"leaf_3_*" + "," + 
					"leaf_4_*" + "," + 
					"tree_*_likelihood" + "," + 
					"leaf_1_i" + "," + 
					"leaf_2_i" + "," + 
					"leaf_3_i" + "," + 
					"leaf_4_i" + "," + 
					"tree_i_likelihood" + "," + 	
					"accept_or_reject" + "," + 
					"ln_r" + "," +
					"ln_u_0_1"
				);			
			TreeIllustration.DeletePreviousTreeIllustrations();
		} catch (IOException e) {
			e.printStackTrace();
		}
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
		
		if (WRITE_DETAILED_DEBUG_FILES){
			for (int t = 0; t < num_trees; t++){
				CGMBARTTreeNode tree = initial_trees.get(t);
				ArrayList<String> all_results = new ArrayList<String>(n);
				for (int i = 0; i < n; i++){
					all_results.add("" + tree.Evaluate(X_y.get(i))); //TreeIllustration.one_digit_format.format(
				} 
				evaluations.println(0 + "," + t + "," + tree.stringID() + "," + Tools.StringJoin(all_results, ","));
			}
		}
	}	
	
	
	public void printTreeIllustations(){
		TREE_ILLUST = true;
	}	
	
	protected static void CloseDebugFiles(){
		tree_liks.close();
		remainings.close();
		sigsqs.close();
		sigsqs_draws.close();
		evaluations.close();
		other_debug.close();		
		mh_iterations_full_record.close();
	}
	
	protected static void OpenDebugFiles(){		
		try {
			sigsqs = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "sigsqs" + DEBUG_EXT, true)));
			other_debug = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "other_debug" + DEBUG_EXT, true)));
			sigsqs_draws = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "sigsqs_draws" + DEBUG_EXT, true)));
			tree_liks = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "tree_liks" + DEBUG_EXT, true)));
			evaluations = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "evaluations" + DEBUG_EXT, true)));
			remainings = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "remainings" + DEBUG_EXT, true)));	
			mh_iterations_full_record = new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_DIR + File.separatorChar + "mh_iterations_full_record" + DEBUG_EXT, true)));
			
		} catch (IOException e) {
			e.printStackTrace();
		}			
	}	

	protected void DebugSample(int sample_num, TreeArrayIllustration tree_array_illustration) {

		if (WRITE_DETAILED_DEBUG_FILES){	
			remainings.println((sample_num) + ",,y," + Tools.StringJoin(y_trans, ","));
			
			ArrayList<CGMBARTTreeNode> current_trees = gibbs_samples_of_cgm_trees.get(sample_num);
			for (int t = 0; t < num_trees; t++){
				CGMBARTTreeNode tree = current_trees.get(t);
				ArrayList<String> all_results = new ArrayList<String>(n);
				for (int i = 0; i < n; i++){
					all_results.add("" + tree.Evaluate(X_y.get(i)));
				}
				evaluations.println(sample_num + "," + t + "," + tree.stringID() + "," + Tools.StringJoin(all_results, ","));
			}	
			evaluations.println((sample_num) + ",,y," + Tools.StringJoin(y_trans, ","));
		}
//		final Thread illustrator_thread = new Thread(){
//			public void run(){
//		if (StatToolbox.rand() < 0.0333){
			if (TREE_ILLUST && sample_num > 3900 && sample_num < 4000 && num_trees == 1){ //
				tree_array_illustration.CreateIllustrationAndSaveImage();
			}
//		}
//			}
//		};
//		illustrator_thread.start();
		
		tree_liks.print("\n");	
		
		sigsqs.println(sample_num + "," + gibbs_samples_of_sigsq.get(sample_num) * y_range_sq);	

		//now close and open all debug
		if (StatToolbox.rand() < 0.0333){
			CloseDebugFiles();
			OpenDebugFiles();
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
	
	public String getRootSplits(int n_g){
		ArrayList<CGMBARTTreeNode> trees = gibbs_samples_of_cgm_trees.get(n_g);
		ArrayList<String> root_splits = new ArrayList<String>(trees.size());
		for (int t = 0; t < trees.size(); t++){
			root_splits.add(trees.get(t).splitToString());
		}
		return Tools.StringJoin(root_splits, "   ||   ");		
	}	

//	protected String treeIDsInCurrentSample(int sample_num){
//		ArrayList<CGMBARTTreeNode> trees = gibbs_samples_of_cgm_trees.get(sample_num);
//		ArrayList<String> treeIds = new ArrayList<String>(trees.size());
//		for (int t = 0; t < trees.size(); t++){
//			treeIds.add(trees.get(t).stringID());
//		}
//		return Tools.StringJoin(treeIds, ",");
//	}


}
