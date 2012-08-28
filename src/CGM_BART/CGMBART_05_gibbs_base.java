package CGM_BART;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

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
		BurnTreeAndSigsqChain();
		//make sure debug files are closed
		CloseDebugFiles();
	}	

	protected void DoGibbsSampling(){	
//		System.out.println("DoGibbsSampling");
		while(gibb_sample_num <= num_gibbs_total_iterations){			
			if (stop_bit){ //rounded to the nearest gibbs sample
				return;
			}	
			if (PrintOutEvery != null && gibb_sample_num % PrintOutEvery == 0){
				System.out.println("gibbs iter: " + gibb_sample_num + "/" + num_gibbs_total_iterations);
			}
			
			DoOneGibbsSampleAndIncrement();
		}
	}
	
	protected void DoOneGibbsSampleAndIncrement(){
		tree_liks.print(gibb_sample_num + ",");
		//this array is the array of trees for this given sample
		final ArrayList<CGMBARTTreeNode> cgm_trees = new ArrayList<CGMBARTTreeNode>(num_trees);				
		final TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(gibb_sample_num, unique_name);
		gibbs_samples_of_cgm_trees.add(null); //so I can set explicitly
		//we cycle over each tree and update it according to formulas 15, 16 on p274
		for (int t = 0; t < num_trees; t++){
			if (t == 0){
				System.out.println("Sampling M_" + (t + 1) + "/" + num_trees + " iter " + gibb_sample_num + "/" + num_gibbs_total_iterations);
			}
			SampleTree(gibb_sample_num, t, cgm_trees, tree_array_illustration);
			SampleMus(gibb_sample_num, t);				
		}
		SampleSigsq(gibb_sample_num);
		DebugSample(gibb_sample_num, tree_array_illustration);
		FlushDataForSample(cgm_trees);
		gibb_sample_num++;
	}

	protected void SampleSigsq(int sample_num) {
		double sigsq = drawSigsqFromPosterior(sample_num);
		gibbs_samples_of_sigsq.add(sample_num, sigsq);
	}

	protected abstract double drawSigsqFromPosterior(int sample_num);

	protected void SampleMus(int sample_num, int t) {
//		System.out.println("SampleMu sample_num " +  sample_num + " t " + t + " gibbs array " + gibbs_samples_of_cgm_trees.get(sample_num));
		CGMBARTTreeNode tree = gibbs_samples_of_cgm_trees.get(sample_num).get(t);
		double current_sigsq = gibbs_samples_of_sigsq.get(sample_num - 1);
		assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(tree, current_sigsq);
	}
	
	protected void SampleTree(int sample_num, int t, ArrayList<CGMBARTTreeNode> cgm_trees, TreeArrayIllustration tree_array_illustration) {
		
		final CGMBARTTreeNode copy_of_old_jth_tree = gibbs_samples_of_cgm_trees.get(sample_num - 1).get(t).clone(true);
//		System.out.println("copy_of_old_jth_tree.data:" + copy_of_old_jth_tree.data + "\n orig_tree.data:" + gibbs_samples_of_cgm_trees.get(sample_num - 1).get(t).data);
//		System.out.println("SampleTreeByCalculatingRemainingsAndDrawingFromTreeDist t:" + t + " of m:" + m);
		//okay so first we need to get "y" that this tree sees. This is defined as R_j
		//in formula 12 on p274
		final double[] R_j = getResidualsBySubtractingTrees(findOtherTrees(sample_num, t));
		
//		System.out.println("SampleTreeByDrawingFromTreeDist rs = " + IOTools.StringJoin(R_j, ","));
		if (WRITE_DETAILED_DEBUG_FILES){
			remainings.println((sample_num - 1) + "," + t + "," + copy_of_old_jth_tree.stringID() + "," + Tools.StringJoin(R_j, ","));			
		}
		
		//now, (important!) set the R_j's as this tree's data.
		copy_of_old_jth_tree.updateWithNewResponsesAndPropagate(X_y, R_j, p);
		
		//sample from T_j | R_j, \sigma
		//now we will run one M-H step on this tree with the y as the R_j
		CGMBARTTreeNode tree_star = metroHastingsPosteriorTreeSpaceIteration(copy_of_old_jth_tree);
		
		//DEBUG
//		System.err.println("tree star: " + tree_star.stringID() + " tree num leaves: " + tree_star.numLeaves() + " tree depth:" + tree_star.deepestNode());
//		double lik = tree_star.
//		tree_liks.print(lik + "," + tree_star.stringID() + ",");
//		tree_array_illustration.addLikelihood(lik);
//		all_tree_liks[t][sample_num] = lik;
		
		cgm_trees.add(t, tree_star);
		//now set the new trees in the gibbs sample pantheon, keep updating it...
		gibbs_samples_of_cgm_trees.set(sample_num, cgm_trees);
//		System.out.println("SampleTree sample_num " + sample_num + " cgm_trees " + cgm_trees);

		tree_array_illustration.AddTree(tree_star);
	}
	
	protected abstract double[] getResidualsBySubtractingTrees(List<CGMBARTTreeNode> findOtherTrees);

	protected abstract ArrayList<CGMBARTTreeNode> findOtherTrees(int sample_num, int t);

	protected abstract CGMBARTTreeNode metroHastingsPosteriorTreeSpaceIteration(CGMBARTTreeNode copy_of_old_jth_tree);

	private void BurnTreeAndSigsqChain() {		
		for (int i = num_gibbs_burn_in; i < num_gibbs_total_iterations; i++){
			gibbs_samples_of_cgm_trees_after_burn_in.add(gibbs_samples_of_cgm_trees.get(i));
			gibbs_samples_of_sigsq_after_burn_in.add(gibbs_samples_of_sigsq.get(i));
		}	
		System.out.println("BurnTreeAndSigsqChain gibbs_samples_of_sigsq_after_burn_in length = " + gibbs_samples_of_sigsq_after_burn_in.size());
	}
	
	public double[] getMuValuesForAllItersByTreeAndLeaf(int t, int leaf_num){
		double[] mu_vals = new double[num_gibbs_total_iterations];
		for (int n_g = 0; n_g < num_gibbs_total_iterations; n_g++){
//			System.out.println("n_g: " + n_g + "length of tree vec: " + gibbs_samples_of_cgm_trees.get(n_g).size());
			CGMBARTTreeNode tree = gibbs_samples_of_cgm_trees.get(n_g).get(t);
			
			Double pred_y = tree.get_pred_for_nth_leaf(leaf_num);
//			System.out.println("t: " + t + " leaf: " + leaf_num + " pred_y: " + pred_y);
			mu_vals[n_g] = un_transform_y(pred_y);
		}
		return mu_vals;
	}	
	
}
