package CGM_BART;


import java.io.Serializable;


public class CGMBARTClassificationMultThread extends CGMBARTRegressionMultThread implements Serializable {
	private static final long serialVersionUID = -3926822473365417428L;

	
	protected void SetupBARTModels() {
//		System.out.print("begin SetupBARTModels()");
		bart_gibbs_chain_threads = new CGMBARTClassification[num_cores];
		for (int t = 0; t < num_cores; t++){
			SetupBartModel(new CGMBARTClassification(), t);
		}
//		System.out.print("end SetupBARTModels()");
	}	
}
