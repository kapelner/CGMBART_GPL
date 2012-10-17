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

import java.util.logging.Logger;
import java.util.logging.StreamHandler;

import CGM_BART_DEBUG.*; //sample #21

public class CGMBARTRegression extends CGMBART_09_eval {
	private static final long serialVersionUID = 6418127647567343927L;
	
	
	/**
	 * Constructs the BART classifier for regression.
	 * 
	 * @param datumSetupForEntireRun
	 * @param buildProgress
	 */
	public CGMBARTRegression() {		
		super();
//		System.out.println("CGMBARTRegression init\n");
		try {
//			Logger.getLogger("").addHandler(new StreamHandler()); //turn off std out
//			suppressOrWriteToDebugLog();
		} 
		catch (Error e){
			System.out.println("Logger and or suppressOrWriteToDebugLog FAILING\n");
		}
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
			output.println(y + "," + yhat + "," + ppi[0] + "," + ppi[1] + "," + inside + "," + Tools.StringJoin(samples, ","));
//			System.out.println(y + "," + yhat + "," + ppi[0] + "," + ppi[1] + "," + inside);
		}
		output.close();
	}
}
