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

import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentTools.IOTools;
import GemIdentView.JProgressBarAndLabel;

public final class CGMBARTClassification extends CGMBART {
	private static final long serialVersionUID = -9061432248755912576L;
	
	/** the number of classes */
//	private Integer K;
	/**
	 * Constructs the BART classifier for classification. We rely on the SetupClassification class to set the raw data
	 * 
	 * @param datumSetup
	 * @param buildProgress
	 */
	public CGMBARTClassification(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress, int K) {
		super(datumSetupForEntireRun, buildProgress);
//		this.K = K;
	}	
	
	@Override
	public double Evaluate(double[] record) {
		return InverseProbit(super.Evaluate(record));
	}	
	

	private double InverseProbit(double y_star) {
		// TODO Auto-generated method stub
		return y_star;
	}
	

	public void writeEvaluationDiagnostics() {
		// TODO Auto-generated method stub
		output.print("y,yhat,a,b,inside");
		for (int i = 0; i < numSamplesAfterBurningAndThinning(); i++){
			output.print(",samp_" + (i + 1));
		}
		output.print("\n");
		for (int i=0; i<n; i++){
			double[] record = X_y.get(i);
			double y = getResponseFromRecord(record);
			double yhat = Evaluate(record);
			double[] ppi = get95PctPostPredictiveIntervalForPrediction(record);
			double[] samples = getGibbsSamplesForPrediction(record);
			int inside = (y >= ppi[0] && y <= ppi[1]) ? 1 : 0;
			output.println(y + "," + yhat + "," + ppi[0] + "," + ppi[1] + "," + inside + "," + IOTools.StringJoin(samples, ","));
		}		
		output.close();
	}

	@Override
	protected void DoGibbsSampling() {
		// TODO Auto-generated method stub
		
	}
}
