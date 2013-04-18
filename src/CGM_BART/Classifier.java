package CGM_BART;

/*
    GemIdent v1.1b
    Interactive Image Segmentation Software via Supervised Statistical Learning
    http://gemident.com
    
    Copyright (C) 2009 Professor Susan Holmes & Adam Kapelner, Stanford University

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


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
//import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.FileHandler;
import java.util.logging.LogManager;
import java.util.logging.Logger;
import java.util.logging.StreamHandler;

import AlgorithmTesting.DataSetupForCSVFile;
import CustomLogging.*;

/**
 * The base class for all machine learning / statistical-learning
 * algorithms. Extend this class to add your own implementation.
 * 
 * @author Adam Kapelner
 */
public abstract class Classifier implements Serializable {
	private static final long serialVersionUID = -2857913059676679308L;	

	public static final double MISSING_VALUE = Double.NaN;

	/** the raw training data consisting of xi = [xi1,...,xiM, yi] that will be used to construct the classifier */
	protected transient ArrayList<double[]> X_y;
	protected transient ArrayList<double[]> X_y_by_col;
	/** just the responses */
	protected transient double[] y_orig;
	protected transient double[] y_trans;
//	/** just the design matrix */
//	protected transient ArrayList<double[]> X;	
	/** the number of records in the training set */
	protected int n;
	/** the number of features / predictors in the training set */
	protected int p;
	
	/** the name of this classifier */
	protected String unique_name = "unnamed";


	private transient double[] in_sample_residuals;
	
	protected static PrintWriter output;
	protected static final String DEBUG_EXT = ".csv";
	
	protected static final String CSVFileFromRName = "bart_data.csv";
	protected static final String CSVFileFromRDirectory = "datasets";		

	
	/** Serializable happy */
	public Classifier(){}
	
	
	public void setDataToDefaultForRPackage(){
		DataSetupForCSVFile data = new DataSetupForCSVFile(new File(CSVFileFromRDirectory, CSVFileFromRName), true);
		setData(data.getX_y());
	}
	
	public void addTrainingDataRow(String[] record_string){
		//initialize data matrix if it hasn't been initialized already
		if (X_y == null){
			X_y = new ArrayList<double[]>();
		}
		
		//now add the new record
		final double[] record = new double[record_string.length];
		for (int i = 0; i < record_string.length; i++){
			try {
				record[i] = Double.parseDouble(record_string[i]);
			}
			catch (NumberFormatException e){
				record[i] = MISSING_VALUE;
				System.out.println("missing value at record #" + X_y.size() + " attribute #" + i);
			}
		}				
		X_y.add(record);		
	}
	
	public void finalizeTrainingData(){
		setData(X_y);
	}
	
	public double[] getResponses(){
		double[] ys = new double[n];
		for (int i = 0; i < n; i++){
			ys[i] = X_y.get(i)[p];
		}
		return ys;
	}	

	
	/** 
	 * adds the data to the classifier - 
	 * data is always a list of int[]'s - call this 
	 * before calling {@link #Build() Build()} 
	 * 
	 * WARNING: It is up to the user to supply raw data with
	 * at least one record, and each record to be in the form
	 * x_i1, x_i2, ..., x_ip, yi
	 * 
	 */
	public void setData(ArrayList<double[]> X_y){
		n = X_y.size();
		p = X_y.get(0).length - 1;
//		System.out.println("setData n:" + n + " p:" + p);
		y_orig = extractResponseFromRawData(X_y);
//		for (int i = 0; i < n; i++){
//			System.out.println("i:" + i + " yi:" + y[i]);
//		}
		transformResponseVariable();
//		X = extractDesignMatrixFromRawData(X_y);
		this.X_y = addIndicesToDataMatrix(X_y);
		this.X_y_by_col = getDataMatrixByCol(X_y);
	}
	
	private ArrayList<double[]> getDataMatrixByCol(ArrayList<double[]> X_y) {
		 ArrayList<double[]> X_y_by_col = new ArrayList<double[]>(n);
		 for (int j = 0; j < p; j++){
			 double[] x_dot_j = new double[n];
			 for (int i = 0; i < n; i++){
				 x_dot_j[i] = X_y.get(i)[j];
			 }
			 X_y_by_col.add(x_dot_j);
		 }
		 return X_y_by_col;
	 }

	private ArrayList<double[]> addIndicesToDataMatrix(ArrayList<double[]> X_y_old) {
		ArrayList<double[]> X_y_new = new ArrayList<double[]>(n);
		for (int i = 0; i < n; i++){
			double[] x = new double[p + 2];
			for (int j = 0; j < p + 1; j++){
				x[j] = X_y_old.get(i)[j];
			}
			x[p + 1] = i;
			X_y_new.add(x);
//			System.out.println("row " + i + ": " + Tools.StringJoin(x));
		}
		return X_y_new;
	}


	private double[] extractResponseFromRawData(ArrayList<double[]> X_y) {
		double[] y = new double[X_y.size()];
		for (int i = 0; i < X_y.size(); i++){
			double[] record = X_y.get(i);
			y[i] = record[record.length - 1];
		}
		return y;
	}
	
	public static ArrayList<double[]> clone_data_matrix_with_new_y_optional(List<double[]> X_y, double[] y_new){
		if (X_y == null){
			return null;
		}
		ArrayList<double[]> X_y_new = new ArrayList<double[]>(X_y.size());
		for (int i = 0; i < X_y.size(); i++){
			double[] original_record = X_y.get(i);
			int num_cols = original_record.length;
			double[] new_record = new double[num_cols];
			for (int j = 0; j < num_cols; j++){
				
				if (j == num_cols - 2 && y_new != null){
//					System.out.println("clone_data_matrix_with_new_y_optional y_new");
					new_record[j] = y_new[i];
				}
				else {
					new_record[j] = original_record[j];
				}
			}
			X_y_new.add(new_record);

//			System.out.println("original_record: " + IOTools.StringJoin(original_record, ","));
//			System.out.println("new_record: " + IOTools.StringJoin(new_record, ","));
		}
		return X_y_new; 
	}	
	
	public ArrayList<double[]> getData() {
		return X_y;
	}	
	
	/** build the machine learning classifier, you must {@link #setData(ArrayList) set the data} first */
	public abstract void Build();
	
	public static void fixRandSeed(){
		StatToolbox.setSeed(1984);
	}
	
	/**
	 * @see https://blogs.oracle.com/nickstephen/entry/java_redirecting_system_out_and
	 */
	public void suppressOrWriteToDebugLog(){
		//also handle the logging
        LogManager logManager = LogManager.getLogManager();
        logManager.reset();

        // create log file, no limit on size
        FileHandler fileHandler = null;
		try {
			fileHandler = new FileHandler(unique_name + ".log", Integer.MAX_VALUE, 1, false);
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
        fileHandler.setFormatter(new SuperSimpleFormatter());
        Logger.getLogger("").addHandler(fileHandler);
        
        
        // now rebind stdout/stderr to logger
        Logger logger = Logger.getLogger("stdout");         
        LoggingOutputStream  los = new LoggingOutputStream(logger, StdOutErrLevel.STDOUT);
        System.setOut(new PrintStream(los, true));
        logger = Logger.getLogger("stderr");                                    
        los = new LoggingOutputStream(logger, StdOutErrLevel.STDERR);            
        System.setErr(new PrintStream(los, true)); 		
	}
	
	/** deletes all data that's unneeded to save memory */
	protected abstract void FlushData();
	
	/** After the classifier has been built, new records can be evaluated */
	public abstract double Evaluate(double[] record, int num_cores);
	
	public double Evaluate(double[] record){
		return Evaluate(record, 1);
	}
	
	/**
	 * Given a data record, return the Y value - take the last index
	 * 
	 * @param record		the data record
	 * @return				its y value (class)
	 */
	public double getResponseFromRecord(double[] record){
		return record[p];
	}

	/** Stop the classifier in its building phase */
	public abstract void StopBuilding();

	public int getP() {
		return p;
	}
	
	public int getN() {
		return n;
	}	
	
	//useful for debugging
	public void dumpDataToFile(String optional_title){
		PrintWriter out=null;
		try {
			out = new PrintWriter(new BufferedWriter(new FileWriter("data_out" + (optional_title == null ? "" : optional_title) + ".csv")));
		} catch (IOException e) {
			System.out.println("cannot be edited in CSV appending");
		}
		
		//print fileheader
		for (int j = 0; j < p; j++){
			out.print("," + j);
		}
		out.print(",y");
		out.print("\n");
		//now print the data
		for (int i = 0; i < n; i++){
			double[] record = X_y.get(i);
			for (int j = 0; j <= p; j++){
				out.print("," + record[j]);
			}
			out.print("\n");
		}
		out.close();		
	}
	
	public static enum ErrorTypes {L1, L2, MISCLASSIFICATION};
	/**
	 * Calculates the in-sample error using the specified loss function
	 * @param num_cores_evaluate 
	 * @param type_of_error_rate  the loss function 
	 * @return the error rate
	 */	
	
	private void calculateInSampleResiduals(int num_cores_evaluate){
		long t0 = System.currentTimeMillis();
		System.out.print("calculating in-sample residuals...");
		in_sample_residuals = new double[n];
		for (int i = 0; i < n; i++){
			double[] record = X_y.get(i);
			double y = getResponseFromRecord(record);
			double yhat = Evaluate(record, num_cores_evaluate);
//			System.out.println("y: " + y + " yhat: " + yhat);
			in_sample_residuals[i] = y - yhat;
		}
		long t1 = System.currentTimeMillis();
		System.out.print("done in " + ((t1 - t0) / 1000.0) + " sec \n");
	}
	
	public double calculateInSampleLoss(ErrorTypes type_of_error_rate, int num_cores_evaluate){	
		if (in_sample_residuals == null){
			calculateInSampleResiduals(num_cores_evaluate);
		}
		
		double loss = 0;
		System.out.print("calculateInSampleLoss for " + type_of_error_rate + "...");
		for (int i = 0; i < n; i++){
			switch (type_of_error_rate){
				case L1:
					loss += Math.abs(in_sample_residuals[i]);
					break;
				case L2:
					loss += in_sample_residuals[i] * in_sample_residuals[i];
					break;
				case MISCLASSIFICATION:
					loss += (in_sample_residuals[i] == 0 ? 0 : 1);
					break;
			}
		}
		System.out.print("done\n");
//		System.out.println("in_sample_residuals: " + Tools.StringJoin(in_sample_residuals));
		return loss;
	}
	
	protected void transformResponseVariable() {
		y_trans = new double[y_orig.length];
		//default is to do nothing... ie just copy the y's into y_trans's
		for (int i = 0; i < n; i++){
			y_trans[i] = y_orig[i];
		}		
	}	
	
	protected double un_transform_y(double y_i) {
		//default:
		return y_i;
	}

	public double calculateMisclassificationRate(int num_cores_evaluate){
		return calculateInSampleLoss(ErrorTypes.MISCLASSIFICATION, num_cores_evaluate) / (double) n * 100;
	}
	
	public double[] getYTrans() {
		return y_trans;
	}		
	
	public Classifier clone(){
		return null;
	}
	
	public void setUniqueName(String unique_name) {
		this.unique_name = unique_name;
	}
	
	public void writeStdOutToLogFile(){
		try {
			Logger.getLogger("").addHandler(new StreamHandler()); //turn off std out
			suppressOrWriteToDebugLog();
		}
		catch (Error e){
			System.out.println("Logger and or suppressOrWriteToDebugLog FAILING\n");
		}		
	}
}
