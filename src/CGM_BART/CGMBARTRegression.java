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

import java.io.File;

import AlgorithmTesting.DataSetupForCSVFile;

import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentTools.IOTools;
import GemIdentView.JProgressBarAndLabel;

public class CGMBARTRegression extends CGMBART1 {
	private static final long serialVersionUID = 6418127647567343927L;
	
	
	/**
	 * Constructs the BART classifier for regression.
	 * 
	 * @param datumSetupForEntireRun
	 * @param buildProgress
	 */
	public CGMBARTRegression(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
		System.out.println("CGMBARTRegression init\n");
	}
	 
	/** Default constructor for R package
	 * 
	 * @param datumSetupForEntireRun
	 * @param buildProgress
	 */
	public CGMBARTRegression() { 
		super(new DataSetupForCSVFile(new File("datasets", "bart_data.csv"), true), new JProgressBarAndLabel(0, 0, null));
		//kind of klunky but whatever 
		this.setData(((DataSetupForCSVFile)this.datumSetupForEntireRun).getX_y());
		System.err.println("default data loaded CGMBARTRegression");
	}	
	 

	public void writeEvaluationDiagnostics() {
		
		output.print("y,yhat,a,b,inside");
		for (int i = 0; i < numSamplesAfterBurningAndThinning(); i++){
			output.print(",samp_" + (i + 1));
		}
		output.print("\n");
		for (int i=0; i<n; i++){
			double[] record = X_y.get(i);
			double y = getResponseFromRecord(record); //the original response from record does not have to be untransformed
			double yhat = Evaluate(record);
			double[] ppi = get95PctPostPredictiveIntervalForPrediction(record);
			double[] samples = getGibbsSamplesForPrediction(record);
			int inside = (y >= ppi[0] && y <= ppi[1]) ? 1 : 0;
			output.println(y + "," + yhat + "," + ppi[0] + "," + ppi[1] + "," + inside + "," + IOTools.StringJoin(samples, ","));
		}		
		output.close();
	}
}
