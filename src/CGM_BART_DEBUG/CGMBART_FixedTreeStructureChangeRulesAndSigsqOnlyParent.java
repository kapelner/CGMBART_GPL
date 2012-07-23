package CGM_BART_DEBUG;

import java.util.ArrayList;

public class CGMBART_FixedTreeStructureChangeRulesAndSigsqOnlyParent extends CGMBART_FixedTreeStructureChangeRulesAndSigsq {
	private static final long serialVersionUID = 3460935328647793073L;	
		
	public CGMBART_FixedTreeStructureChangeRulesAndSigsqOnlyParent() {
		super();
		System.out.println("CGMBART_Alt init\n");
		setNumTrees(1); //in this DEBUG model, there's only one tree
//		printTreeIllustations();
	}
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		fixed_sigsq = 1 / y_range_sq;
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
