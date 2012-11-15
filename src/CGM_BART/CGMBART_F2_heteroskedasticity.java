package CGM_BART;


public class CGMBART_F2_heteroskedasticity extends CGMBART_F1_prior_cov_spec {
	private static final long serialVersionUID = -3069428133597923502L;

	protected boolean use_heteroskedasticity;
	
	
	private double calcLnLikRatioGrowF2(CGMBARTTreeNode grow_node) {
		// TODO Auto-generated method stub
		return 0;
	}	
	
	private void SampleSigsqF2(int sample_num, double[] r_j) {
		// TODO Auto-generated method stub
		
	}	
	
	private void SampleMusF2(int sample_num, int t) {
		
	}	
	

	public void useHeteroskedasticity(){
		use_heteroskedasticity = true;
	}	
	
	
	/////////////nothing but scaffold code below, do not alter!
	
	protected void SampleMus(int sample_num, int t) {
		if (use_heteroskedasticity){
			SampleMusF2(sample_num, t);
		}
		else {
			super.SampleMus(sample_num, t);
		}
	}	

	protected void SampleSigsq(int sample_num, double[] R_j) {
		if (use_heteroskedasticity){
			SampleSigsqF2(sample_num, R_j);
		}
		else {
			super.SampleSigsq(sample_num, R_j);
		}		
	}

	protected double calcLnLikRatioGrow(CGMBARTTreeNode grow_node) {
		if (use_heteroskedasticity){
			return calcLnLikRatioGrowF2(grow_node);
		}
		return super.calcLnLikRatioGrow(grow_node);
	}	
	
}
