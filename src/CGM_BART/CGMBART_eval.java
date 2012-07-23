package CGM_BART;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import CGM_Statistics.StatToolbox;

public abstract class CGMBART_eval extends CGMBART_mh implements Serializable {
	private static final long serialVersionUID = -6670611007413531590L;

	public double Evaluate(double[] record) { //posterior sample median (it's what Rob uses)		
		return EvaluateViaSampMed(record);
	}	
	
	public double EvaluateViaSampMed(double[] record) { //posterior sample average		
		return StatToolbox.sample_median(getGibbsSamplesForPrediction(record));
	}
	
	public double EvaluateViaSampAvg(double[] record) { //posterior sample average		
		return StatToolbox.sample_average(getGibbsSamplesForPrediction(record));
	}

	protected double[] getGibbsSamplesForPrediction(double[] record){
//		System.out.println("eval record: " + record + " numtrees:" + this.bayesian_trees.size());
		//the results for each of the gibbs samples
		double[] y_gibbs_samples = new double[numSamplesAfterBurningAndThinning()];	
		for (int i = 0; i < numSamplesAfterBurningAndThinning(); i++){
			ArrayList<CGMBARTTreeNode> cgm_trees = gibbs_samples_of_cgm_trees_after_burn_in.get(i);
			double yt_i = 0;
			for (CGMBARTTreeNode tree : cgm_trees){ //sum of trees right?
				yt_i += tree.Evaluate(record);
			}
			//just make sure we switch it back to really what y is for the user
			y_gibbs_samples[i] = un_transform_y(yt_i);
		}
		return y_gibbs_samples;
	}
	
	protected double[] getPostPredictiveIntervalForPrediction(double[] record, double coverage){
		//get all gibbs samples sorted
		double[] y_gibbs_samples_sorted = getGibbsSamplesForPrediction(record);
		Arrays.sort(y_gibbs_samples_sorted);
		
		//calculate index of the CI_a and CI_b
		int n_bottom = (int)Math.round((1 - coverage) / 2 * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
		int n_top = (int)Math.round(((1 - coverage) / 2 + coverage) * y_gibbs_samples_sorted.length) - 1; //-1 because arrays start at zero
//		System.out.print("getPostPredictiveIntervalForPrediction record = " + IOTools.StringJoin(record, ",") + "  Ng=" + y_gibbs_samples_sorted.length + " n_a=" + n_bottom + " n_b=" + n_top + " guess = " + Evaluate(record));
		double[] conf_interval = {y_gibbs_samples_sorted[n_bottom], y_gibbs_samples_sorted[n_top]};
//		System.out.println("  [" + conf_interval[0] + ", " + conf_interval[1] + "]");
		return conf_interval;
	}
	
	protected double[] get95PctPostPredictiveIntervalForPrediction(double[] record){
		return getPostPredictiveIntervalForPrediction(record, 0.95);
	}
}
