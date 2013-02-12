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

public final class CGMBARTClassification extends CGMBARTRegression {
	private static final long serialVersionUID = -9061432248755912576L;
	private static final double SIGSQ_FOR_PROBIT = 1;


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
	protected void DoOneGibbsSample(){
//		System.out.println("DoOneGibbsSample CGMBARTClassification");
		//this array is the array of trees for this given sample
		final CGMBARTTreeNode[] cgm_trees = new CGMBARTTreeNode[num_trees];				
		final TreeArrayIllustration tree_array_illustration = new TreeArrayIllustration(gibbs_sample_num, unique_name);

		//we cycle over each tree and update it according to formulas 15, 16 on p274
		for (int i = 0; i < n; i++){
			double g_x_i = 0;
			CGMBARTTreeNode[] trees = gibbs_samples_of_cgm_trees[gibbs_sample_num - 1];
			for (int t = 0; t < num_trees; t++){
				g_x_i += trees[t].Evaluate(X_y.get(i));
			}
			//y_trans is the Z's from the paper
			y_trans[i] = SampleZi(g_x_i, y_orig[i]);
		}
//		System.out.println("SampleZis: " + Tools.StringJoin(y_trans));
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
	
	private double SampleZi(double g_x_i, double y_i) {
		if (y_i == 1){
			return Math.max(StatToolbox.sample_from_norm_dist(g_x_i, SIGSQ_FOR_PROBIT), 0);
		} 
		else if (y_i == 0){
			return Math.min(StatToolbox.sample_from_norm_dist(g_x_i, SIGSQ_FOR_PROBIT), 0);
		}
		System.err.println("SampleZi RESPONSE NOT ZERO / ONE");
		System.exit(0);
		return -1;
	}

	protected void SetupGibbsSampling(){
		super.SetupGibbsSampling();
		//all sigsqs are now 1 all the time
		for (int g = 0; g < num_gibbs_total_iterations; g++){
			gibbs_samples_of_sigsq[g] = SIGSQ_FOR_PROBIT;
		}
	}

	//all we need is the new sigsq_mu hyperparam
	protected void calculateHyperparameters() {
//		System.out.println("calculateHyperparameters in CGMBARTClassification\n\n");
		hyper_mu_mu = 0;
		hyper_sigsq_mu = Math.pow(3 / (hyper_k * Math.sqrt(num_trees)), 2);	
	}
	
	
	//do nothing
	protected void transformResponseVariable() {
		y_trans = new double[y_orig.length];		
	}	
	
	//do nothing
	public double un_transform_y(double yt_i){
		return yt_i;
	}	
}
