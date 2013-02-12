/*
    BART - Bayesian Additive Regressive Trees
    Software for Supervised Statistical Learning
    
    Copyright (C) 2012 Professor Ed George & Adam Kapelner, 
    Dept of Statistics, The Wharton School of the University of Pennsylvania

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details:
    
    http://www.gnu.org/licenses/gpl-2.0.txt

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

package CGM_BART;

import java.util.ArrayList;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;

public final class CGMBARTClassification extends CGMBARTRegression {
	private static final long serialVersionUID = -9061432248755912576L;
	private static final double SIGSQ_FOR_PROBIT = 1;
	
	private double[] current_zs;
	
	/**
	 * Constructs the BART classifier for classification. We rely on the SetupClassification class to set the raw data
	 * 
	 * @param datumSetup
	 * @param buildProgress
	 */
	public CGMBARTClassification() {
		super();
		
	}	
	
	@Override
	public double Evaluate(double[] record) {
		return InverseProbit(super.Evaluate(record));
	}	
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		current_zs = new double[n];
	}
	
	
	private double InverseProbit(double y_star) {
		// TODO Auto-generated method stub
		return y_star;
	}
		

	@Override
	protected void DoOneGibbsSample(){
//		tree_liks.print(gibb_sample_num + ",");
		//this array is the array of trees for this given sample
		final CGMBARTTreeNode[] cgm_trees = new CGMBARTTreeNode[num_trees];				
		final TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(gibbs_sample_num, unique_name);

		//we cycle over each tree and update it according to formulas 15, 16 on p274
		for (int i = 0; i < n; i++){
			double g_x_i = 0;
			for (int t = 0; t < num_trees; t++){
				g_x_i += gibbs_samples_of_cgm_trees[gibbs_sample_num - 1][t].Evaluate(X_y.get(i));
			}
			current_zs[i] = SampleZi(g_x_i, y_orig[i]);
		}
		for (int t = 0; t < num_trees; t++){
			if (t == 0 && gibbs_sample_num % 100 == 0){
				//debug memory messages
				long mem_used = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
				long max_mem = Runtime.getRuntime().maxMemory();
				System.out.println("Sampling M_" + (t + 1) + "/" + num_trees + " iter " + 
						gibbs_sample_num + "/" + num_gibbs_total_iterations + "  thread: " + (threadNum + 1) +
						"  mem: " + TreeIllustration.one_digit_format.format(mem_used / 1000000.0) + "/" + 
						TreeIllustration.one_digit_format.format(max_mem / 1000000.0) + "MB");
			}
			SampleTree(gibbs_sample_num, t, cgm_trees, tree_array_illustration);
			SampleMusWrapper(gibbs_sample_num, t);				
		}
	}
	
	private double SampleZi(double g_x_i, double d) {
		// TODO Auto-generated method stub
		return 0;
	}

	protected void SetupGibbsSampling(){
		super.SetupGibbsSampling();
		//we no longer care about sigsq's so let's do defaults:
		for (int g = 0; g < num_gibbs_total_iterations; g++){
			gibbs_samples_of_sigsq[g] = SIGSQ_FOR_PROBIT;
		}
	}

	protected void calculateHyperparameters() {
//		System.out.println("calculateHyperparameters in BART\n\n");
		hyper_mu_mu = 0;
		hyper_sigsq_mu = Math.pow(3 / (hyper_k * Math.sqrt(num_trees)), 2);
	
	}
	
	
	protected void transformResponseVariable() {
		y_trans = new double[y_orig.length];
		//default is to do nothing... ie just copy the y's into y_trans's
		for (int i = 0; i < n; i++){
			y_trans[i] = y_orig[i];
		}		
	}	
}
