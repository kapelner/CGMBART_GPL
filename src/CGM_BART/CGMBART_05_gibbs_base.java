package CGM_BART;

import java.io.Serializable;

public abstract class CGMBART_05_gibbs_base extends CGMBART_04_init implements Serializable {
	private static final long serialVersionUID = 1280579612167425306L;

	
	@Override
	public void Build() {
//		System.out.println("Build CGMBART_05_gibbs_base");
		//this can be different for any BART implementation
		SetupGibbsSampling();		
		//this section is different for the different BART implementations
		//but it basically does all the Gibbs sampling
		DoGibbsSampling();
		//now we burn and thin the chains for each param
//		BurnTreeAndSigsqChain();
		//make sure debug files are closed
//		CloseDebugFiles();
	}	

	protected void DoGibbsSampling(){	
//		System.out.println("DoGibbsSampling");
		while(gibbs_sample_num <= num_gibbs_total_iterations){			
			if (stop_bit){ //rounded to the nearest gibbs sample
				return;
			}	
//			if (PrintOutEvery != null && gibb_sample_num % PrintOutEvery == 0){
//				System.out.println("gibbs iter: " + gibb_sample_num + "/" + num_gibbs_total_iterations);
//			}
			
			DoOneGibbsSample();
			//now flush the previous previous gibbs sample to not leak memory
			FlushDataForSample(gibbs_samples_of_cgm_trees[gibbs_sample_num - 1]);
			DeleteBurnInSampleOnOtherThreads();
//			System.gc();
			//debug memory messages
			if (gibbs_sample_num % 100 == 0){
				long mem_used = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
				long max_mem = Runtime.getRuntime().maxMemory();
				System.out.println(" mem_used = " + mem_used / 1000000.0 + "MB" + 
						" max_mem = " + max_mem / 1000000.0 + "MB" + "  thread: " + (threadNum + 1));
			}
			gibbs_sample_num++;					
		}
	}
	
	protected void DoOneGibbsSample(){
//		tree_liks.print(gibb_sample_num + ",");
		//this array is the array of trees for this given sample
		final CGMBARTTreeNode[] cgm_trees = new CGMBARTTreeNode[num_trees];				
		final TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(gibbs_sample_num, unique_name);

		///////////////NO NEED FOR THIS HACK ANYMORE
		//		gibbs_samples_of_cgm_trees.add(null); //so I can set explicitly
		//we cycle over each tree and update it according to formulas 15, 16 on p274
		double[] R_j = new double[n];
		for (int t = 0; t < num_trees; t++){
			if (t == 0 && gibbs_sample_num % 100 == 0){				
				System.out.println("Sampling M_" + (t + 1) + "/" + num_trees + " iter " + 
					gibbs_sample_num + "/" + num_gibbs_total_iterations + "  thread: " + (threadNum + 1));
			}
			R_j = SampleTree(gibbs_sample_num, t, cgm_trees, tree_array_illustration);
			SampleMus(gibbs_sample_num, t);				
		}
		//now we have the last residual vector which we pass on to sample sigsq
		SampleSigsq(gibbs_sample_num, R_j);
		DebugSample(gibbs_sample_num, tree_array_illustration);
	}

	private void DeleteBurnInSampleOnOtherThreads() {
		if (threadNum > 0 && gibbs_sample_num <= num_gibbs_burn_in + 1 && gibbs_sample_num >= 2){
			gibbs_samples_of_cgm_trees[gibbs_sample_num - 2] = null;
//			System.out.println("DeleteBurnInSampleOnOtherThreads() thread:" + (threadNum + 1) + " gibbs_sample_num = " + gibbs_sample_num + " num_gibbs_burn_in = " + num_gibbs_burn_in + " len = " + gibbs_samples_of_cgm_trees.size());
			
		}
	}

	protected void SampleSigsq(int sample_num, double[] R_j) {
		double sigsq = drawSigsqFromPosterior(sample_num, R_j);
		gibbs_samples_of_sigsq[sample_num] = sigsq;
	}

	protected abstract double drawSigsqFromPosterior(int sample_num, double[] R_j);

	protected void SampleMus(int sample_num, int t) {
		CGMBARTTreeNode previous_tree = gibbs_samples_of_cgm_trees[sample_num - 1][t];
		//subtract out previous tree's yhats
//		System.out.println("  previous yhats = " + Tools.StringJoin(previous_tree.yhats));
		sum_resids_vec = Tools.subtract_arrays(sum_resids_vec, previous_tree.yhats);
//		System.out.println("SampleMu sample_num " +  sample_num + " t " + t + " gibbs array " + gibbs_samples_of_cgm_trees.get(sample_num));
		CGMBARTTreeNode tree = gibbs_samples_of_cgm_trees[sample_num][t];
		double current_sigsq = gibbs_samples_of_sigsq[sample_num - 1];
		assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqAndUpdateYhats(tree, current_sigsq);
		//after mus are sampled, we need to update the sum_resids_vec
		//add in current tree's yhats
//		System.out.println("  current  yhats = " + Tools.StringJoin(tree.yhats));
		
		sum_resids_vec = Tools.add_arrays(sum_resids_vec, tree.yhats);
	}
	
	protected double[] SampleTree(int sample_num, int t, CGMBARTTreeNode[] cgm_trees, TreeArrayIllustration tree_array_illustration) {
		//first copy the tree from the previous gibbs position
		final CGMBARTTreeNode copy_of_old_jth_tree_root = gibbs_samples_of_cgm_trees[sample_num - 1][t].clone();
		
		//okay so first we need to get "y" that this tree sees. This is defined as R_j in formula 12 on p274
		//just go to sum_residual_vec and subtract it from y_trans
		double[] R_j = Tools.add_arrays(Tools.subtract_arrays(y_trans, sum_resids_vec), copy_of_old_jth_tree_root.yhats);
//		System.out.println("sample tree gs#" + sample_num + " t = " + t + " sum_resids = " + Tools.StringJoin(sum_resids_vec));
		
		//now, (important!) set the R_j's as this tree's data.
		copy_of_old_jth_tree_root.updateWithNewResponsesRecursively(R_j);
		
		//sample from T_j | R_j, \sigma
		//now we will run one M-H step on this tree with the y as the R_j
		CGMBARTTreeNode new_jth_tree = metroHastingsPosteriorTreeSpaceIteration(copy_of_old_jth_tree_root);
		
		//add it to the vector of current sample's trees
		cgm_trees[t] = new_jth_tree;
		
		//now set the new trees in the gibbs sample pantheon, keep updating it...
		gibbs_samples_of_cgm_trees[sample_num] = cgm_trees;
		tree_array_illustration.AddTree(new_jth_tree);		
		return R_j;
	}
	
	protected abstract CGMBARTTreeNode metroHastingsPosteriorTreeSpaceIteration(CGMBARTTreeNode copy_of_old_jth_tree);

//	private void BurnTreeAndSigsqChain() {		
//		for (int i = num_gibbs_burn_in; i < num_gibbs_total_iterations; i++){
//			gibbs_samples_of_cgm_trees_after_burn_in[i - num_gibbs_burn_in] = gibbs_samples_of_cgm_trees[i];
//			gibbs_samples_of_sigsq_after_burn_in[i - num_gibbs_burn_in] = gibbs_samples_of_sigsq[i];
//		}	
////		System.out.println("BurnTreeAndSigsqChain gibbs_samples_of_sigsq_after_burn_in length = " + gibbs_samples_of_sigsq_after_burn_in.size());
//	}
	
//	public double[] getMuValuesForAllItersByTreeAndLeaf(int t, int leaf_num){
//		double[] mu_vals = new double[num_gibbs_total_iterations];
//		for (int n_g = 0; n_g < num_gibbs_total_iterations; n_g++){
////			System.out.println("n_g: " + n_g + "length of tree vec: " + gibbs_samples_of_cgm_trees.get(n_g).size());
//			CGMBARTTreeNode tree = gibbs_samples_of_cgm_trees.get(n_g).get(t);
//			
//			Double pred_y = tree.get_pred_for_nth_leaf(leaf_num);
////			System.out.println("t: " + t + " leaf: " + leaf_num + " pred_y: " + pred_y);
//			mu_vals[n_g] = un_transform_y(pred_y);
//		}
//		return mu_vals;
//	}	
	
	public double[] getAvgCountsByAttribute(){
		double[] avg_counts = new double[p];
		for (int j = 0; j < p; j++){
			int tot_for_attr_j = 0;
			for (int g = 0; g < numSamplesAfterBurningAndThinning(); g++){
				for (CGMBARTTreeNode root_node : gibbs_samples_of_cgm_trees_after_burn_in[g]){
					tot_for_attr_j += root_node.numTimesAttrUsed(j);
				}		
			}			
			avg_counts[j] = tot_for_attr_j / (double)numSamplesAfterBurningAndThinning();
		}
		
		return avg_counts;
	}

}
