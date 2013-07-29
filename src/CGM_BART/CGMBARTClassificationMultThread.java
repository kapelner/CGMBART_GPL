package CGM_BART;


import java.io.Serializable;


public class CGMBARTClassificationMultThread extends CGMBARTRegressionMultThread implements Serializable {
	private static final long serialVersionUID = -3926822473365417428L;
	
	private static double DEFAULT_CLASSIFICATION_RULE = 0.5;
	private double classification_rule;

	
	protected void SetupBARTModels() {
//		System.out.print("begin SetupBARTModels()");
		bart_gibbs_chain_threads = new CGMBARTClassification[num_cores];
		for (int t = 0; t < num_cores; t++){
			SetupBartModel(new CGMBARTClassification(), t);
		}
//		System.out.print("end SetupBARTModels()");
		classification_rule = DEFAULT_CLASSIFICATION_RULE;
	}
	
	public double Evaluate(double[] record, int num_cores_evaluate) { //posterior sample median (it's what Rob uses)		
//		System.out.println("Evaluate CGMBARTClassificationMultThread");
		return EvaluateViaSampAvg(record, num_cores_evaluate) > classification_rule ? 1 : 0;
	}	
	
	protected double[][] getGibbsSamplesForPrediction(double[][] data, int num_cores_evaluate){
		double[][] y_gibbs_samples = super.getGibbsSamplesForPrediction(data, num_cores_evaluate);
		double[][] y_gibbs_samples_probs = new double[y_gibbs_samples.length][y_gibbs_samples[0].length];
		for (int g = 0; g < y_gibbs_samples.length; g++){
			for (int i = 0; i < y_gibbs_samples[0].length; i++){
				y_gibbs_samples_probs[g][i] = StatToolbox.normal_cdf(y_gibbs_samples[g][i]);
			}			
		}
//		System.out.println("y_gibbs_samples_probs: " + Tools.StringJoin(y_gibbs_samples_probs[0]));
		return y_gibbs_samples_probs;
	}	
	
	public void setClassificationRule(double classification_rule) {
		this.classification_rule = classification_rule;
	}	
	
	public void useLinearHeteroskedasticityModel(){} //cannot let that flag get set in the father class since there is no noise in a probit model
}
