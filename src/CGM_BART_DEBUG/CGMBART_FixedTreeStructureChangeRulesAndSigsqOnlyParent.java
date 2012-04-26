package CGM_BART_DEBUG;

import java.util.ArrayList;

import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentView.JProgressBarAndLabel;
import CGM_BART.*;
import CGM_Statistics.CGMTreeNode;

public class CGMBART_FixedTreeStructureChangeRulesAndSigsqOnlyParent extends CGMBART_FixedTreeStructureChangeRulesAndSigsq {
	private static final long serialVersionUID = 3460935328647793073L;
	private static boolean TREE_ILLUST = true;
		
	public CGMBART_FixedTreeStructureChangeRulesAndSigsqOnlyParent(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
		System.out.println("CGMBART_Alt init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
//		printTreeIllustations();
	}
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		//this posterior builder will be shared throughout the entire process
		posterior_builder = new CGMBARTPosteriorBuilder_OnlyParent(tree_prior_builder);
		//we have to set the CGM98 hyperparameters as well as the hyperparameter sigsq_mu
		posterior_builder.setHyperparameters(hyper_mu_mu, hyper_sigsq_mu);
		//set sigsq
		fixed_sigsq = 1 / y_range_sq;
		posterior_builder.setCurrentSigsqValue(fixed_sigsq);
	}
//	
//	//fix it once for good
//	protected void InitizializeSigsq() {		
//		gibbs_samples_of_sigsq.add(0, fixed_sigsq);		
//	}
//	
//	protected void SampleSigsq(int sample_num) {
//		gibbs_samples_of_sigsq.add(sample_num, fixed_sigsq); //fix it forever
//	}	
	
}
