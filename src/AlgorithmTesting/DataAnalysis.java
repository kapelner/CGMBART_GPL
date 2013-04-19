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

package AlgorithmTesting;

import java.io.File;
import java.io.IOException;

import CGM_BART.*;

public class DataAnalysis {
	
	/** this is a file that is in CSV format (csv extension) with/out a header named c_<name> or r_<name> for classification or regression respectively */
//	private static final String DataSetFilename = "r_just_noise";
//	private static final String DataSetFilename = "r_treemodel";
//	private static final String DataSetFilename = "r_treemodel_high_p";
//	private static final String DataSetFilename = "r_treemodel_high_p_low_n";
//	private static final String DataSetFilename = "r_treemodel_high_n";
//	private static final String DataSetFilename = "r_treemodel_low_n";	
//	private static final String DataSetFilename = "r_friedman";
//	private static final String DataSetFilename = "r_friedman_hd";	
//	private static final String DataSetFilename = "r_univariatelinear";
//	private static final String DataSetFilename = "r_bivariatelinear";
//	private static final String DataSetFilename = "r_boston";
//	private static final String DataSetFilename = "r_boston_half";	
	private static final String DataSetFilename = "r_boston_tiny_with_missing";	
//	private static final String DataSetFilename = "r_zach";
//	private static final String DataSetFilename = "r_forestfires";
//	private static final String DataSetFilename = "r_wine_white";
//	private static final String DataSetFilename = "r_concretedata";
//	private static final String DataSetFilename = "c_breastcancer";
//	private static final String DataSetFilename = "c_crime";
//	private static final String DataSetFilename = "c_crime_big";	

	public static void main(String[] args) throws IOException{
		System.out.println("java ver: " + System.getProperty("java.version"));
		//make sure y is last column of data matrix
		DataSetupForCSVFile data = new DataSetupForCSVFile(new File("datasets", DataSetFilename + ".csv"), true);
		Classifier machine = null; //we're going to use some machine to do it... 
		
		//if the filename begins with a "c" => classification task, if it begins with an "r" => regression task
		if (DataSetFilename.charAt(0) == 'c'){ //classification problem
//			CGMClassificationTree tree = new CGMClassificationTree(data, new JProgressBarAndLabel(0, 0, null), data.getK());
			machine = new CGMBARTClassificationMultThread();
			long start_time = System.currentTimeMillis();
			machine.setData(data.getX_y());
			
			machine.Build(); 
			System.out.println("errors: " + 
					(int)machine.calculateInSampleLoss(Classifier.ErrorTypes.MISCLASSIFICATION, 4) + 
					"/" + 
					machine.getN() + 
					"  (" + machine.calculateMisclassificationRate(4) + "%)");
			long end_time = System.currentTimeMillis();
			System.out.println("Current Time:"+ (end_time-start_time)/1000);
		}
		
		else {
		//regression problem
//			machine = new RandomForest(data, new JProgressBarAndLabel(0, 0, null));
//			for (int num_times = 0; num_times < 100; num_times++){
			machine = new CGMBARTRegressionMultThread();
			machine.setData(data.getX_y());
//				double[] cov_split_prior = {1, 1000,1000};
//				((CGMBARTRegressionMultThread)machine).setCovSplitPrior(cov_split_prior);
//				((CGMBARTRegressionMultThread)machine).useHeteroskedasticity();
			machine.Build();
			long L2 = Math.round(machine.calculateInSampleLoss(Classifier.ErrorTypes.L2, 4));
			System.out.println("(in sample) L1 error: " + 
				Math.round(machine.calculateInSampleLoss(Classifier.ErrorTypes.L1, 4)) +
				" L2 error: " + L2 + " rmse: " + Math.sqrt(L2 / (double)machine.getN()));
		}
	}
}
