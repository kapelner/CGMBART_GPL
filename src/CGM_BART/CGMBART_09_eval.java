package CGM_BART;

import java.io.Serializable;
import java.util.Arrays;


public abstract class CGMBART_09_eval extends CGMBART_07_mh implements Serializable {
	private static final long serialVersionUID = -6670611007413531590L;

	public double Evaluate(double[] record, int num_cores_evaluate) { //posterior sample median (it's what Rob uses)		
		return EvaluateViaSampAvg(record, num_cores_evaluate);
	}	
	
	public double EvaluateViaSampMed(double[] record, int num_cores_evaluate) { //posterior sample average		
		return StatToolbox.sample_median(getGibbsSamplesForPrediction(record, num_cores_evaluate));
	}
	
	public double EvaluateViaSampAvg(double[] record, int num_cores_evaluate) { //posterior sample average		
		return StatToolbox.sample_average(getGibbsSamplesForPrediction(record, num_cores_evaluate));
	}

	protected double[] getGibbsSamplesForPrediction(double[] data_record, int num_cores_evaluate){
//		System.out.println("eval record: " + record + " numtrees:" + this.bayesian_trees.size());
		//the results for each of the gibbs samples
		double[] y_gibbs_samples = new double[numSamplesAfterBurningAndThinning()];	
		for (int g = 0; g < numSamplesAfterBurningAndThinning(); g++){
			CGMBARTTreeNode[] cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in[g];
			double yt_g = 0;
			for (CGMBARTTreeNode tree : cgm_trees){ //sum of trees right?
				yt_g += tree.Evaluate(data_record);
			}			
			y_gibbs_samples[g] = un_transform_y(yt_g);
		}
		return y_gibbs_samples;
	}
	
	protected double[] getPostPredictiveIntervalForPrediction(double[] record, double coverage, int num_cores_evaluate){
		//get all gibbs samples sorted
		double[] y_gibbs_samples_sorted = getGibbsSamplesForPrediction(record, num_cores_evaluate);
		Arrays.sort(y_gibbs_samples_sorted);
		
		//calculate index of the CI_a and CI_b
		int n_bottom = (int)Math.round((1 - coverage) / 2 * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
		int n_top = (int)Math.round(((1 - coverage) / 2 + coverage) * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
//		System.out.print("getPostPredictiveIntervalForPrediction record = " + IOTools.StringJoin(record, ",") + "  Ng=" + y_gibbs_samples_sorted.length + " n_a=" + n_bottom + " n_b=" + n_top + " guess = " + Evaluate(record));
		double[] conf_interval = {y_gibbs_samples_sorted[n_bottom], y_gibbs_samples_sorted[n_top]};
//		System.out.println("  [" + conf_interval[0] + ", " + conf_interval[1] + "]");
		return conf_interval;
	}
	
	protected double[] get95PctPostPredictiveIntervalForPrediction(double[] record, int num_cores_evaluate){
		return getPostPredictiveIntervalForPrediction(record, 0.95, num_cores_evaluate);
	}	
}
