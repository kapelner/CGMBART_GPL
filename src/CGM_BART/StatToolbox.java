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

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.GammaDistributionImpl;
import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.apache.commons.math.special.Gamma;

/**
 * This is a class where we're going to put all sorts of useful functions
 * as a utility-style class
 */
public class StatToolbox {
	
	public static final Random R = new Random();
	
	public static double rand(){
		return R.nextDouble();
	}

	/**
	 * Wikipedia parameterization
	 */
	public static final double ILLEGAL_FLAG = -999999999;

	public static double sample_from_inv_gamma(double k, double theta){
//		System.out.println("sample_from_inv_gamma k = " + k + " theta = " + theta);
		return 1 / sample_from_gamma(k, theta);
	}
	
	public static double sample_from_gamma(double k, double theta){
		GammaDistributionImpl gamma_dist = new GammaDistributionImpl(k, theta);
		try {
			return gamma_dist.inverseCumulativeProbability(StatToolbox.rand());
		} catch (MathException e) {
//			System.out.println("sample_from_inv_gamma failed: " + e.toString());
			e.printStackTrace();
		}
		return ILLEGAL_FLAG;
	}
	
	public static double inv_norm_dist(double p){
		//System.out.println("inv_norm_dist p=" + p);
		try {
			return new NormalDistributionImpl().inverseCumulativeProbability(p);
		} catch (MathException e) {
			e.printStackTrace();
		}
		return ILLEGAL_FLAG;
	}
	
	private static double[] NORM_SAMPS;
	private static int NUM_NORM_SAMPS;
	private static int START_POS;
	private static final String norm_samps_file = "randsamps/rnorm.csv";
	static {
		BufferedReader in;
		try {
			in = new BufferedReader(new FileReader(norm_samps_file));
			try {
				String raw = in.readLine();
				String[] random_samps = raw.split(",");
				NUM_NORM_SAMPS = random_samps.length;
				NORM_SAMPS = new double[NUM_NORM_SAMPS];
				START_POS = (int)Math.floor(rand() * NUM_NORM_SAMPS);
				for (int i = 0; i < NUM_NORM_SAMPS; i++){
					NORM_SAMPS[i] = Double.parseDouble(random_samps[i]);
				}	
				System.out.println("NUM_NORM_SAMPS: " + NUM_NORM_SAMPS);
//				System.out.println("START_POS: " + START_POS);
//				System.out.println("NORM_SAMPS: " + Tools.StringJoin(NORM_SAMPS));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}			
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
					
	}
	
	public static double sample_from_norm_dist(double mu, double sigsq){
		double std_norm_realization = NORM_SAMPS[(int)Math.floor(rand() * NUM_NORM_SAMPS)];
//		System.out.println("sample_from_norm_dist S = " + START_POS + " P = " + START_POS % NUM_NORM_SAMPS + "real = " + std_norm_realization);
//		START_POS++;
		return mu + Math.sqrt(sigsq) * std_norm_realization;
	}	
	
//	public static double sample_from_norm_dist(double mu, double sigsq){
////		System.out.println("sample_from_norm_dist mu=" + mu + " sigsq=" + sigsq);
//		try {
//			return new NormalDistributionImpl(mu, Math.sqrt(sigsq)).inverseCumulativeProbability(StatToolbox.rand());
//		} catch (MathException e) {
////			System.err.println("ERROR sample_from_norm_dist mu=" + mu + " sigsq=" + sigma);
////			e.printStackTrace();
//		}
//		return ILLEGAL_FLAG;		
//	}
	
	// test the sampling of the normal
//	static {
//		for (int i = 0; i < 1000; i++){
//			System.out.println(sample_from_norm_dist(0, 2));
//		}
//	}
	
	public static final double sample_average(double[] y){
		double y_bar = 0;
		for (int i = 0; i < y.length; i++){
			y_bar += y[i];
		}
		return y_bar / (double)y.length;
	}
	
	public static final double sample_average(TDoubleArrayList y){
		double y_bar = 0;
		for (int i = 0; i < y.size(); i++){
			y_bar += y.get(i);
		}
		return y_bar / (double)y.size();
	}	
	
	public static final double sample_average(int[] y){
		double y_bar = 0;
		for (int i = 0; i < y.length; i++){
			y_bar += y[i];
		}
		return y_bar / (double)y.length;
	}	
	
	public static final double sample_standard_deviation(int[] y){
		double y_bar = sample_average(y);
		double sum_sqd_deviations = 0;
		for (int i = 0; i < y.length; i++){
			sum_sqd_deviations += Math.pow(y[i] - y_bar, 2);
		}
		return Math.sqrt(sum_sqd_deviations / ((double)y.length - 1));		
	}
	
	public static final double sample_standard_deviation(double[] y){
		return Math.sqrt(sample_variance(y));
	}	
	
	public static final double sample_variance(double[] y){
		return sample_sum_sq_err(y) / ((double)y.length - 1);		
	}		
	
	public static final double sample_sum_sq_err(double[] y){
		double y_bar = sample_average(y);
		double sum_sqd_deviations = 0;
		for (int i = 0; i < y.length; i++){
			sum_sqd_deviations += Math.pow(y[i] - y_bar, 2);
		}
		return sum_sqd_deviations;
	}
	
	public static final double cumul_dens_function_inv_gamma(double alpha, double beta, double lower, double upper){
		return cumul_dens_function_inv_gamma(alpha, beta, upper) - cumul_dens_function_inv_gamma(alpha, beta, lower);
	}
	
	public static final double cumul_dens_function_inv_gamma(double alpha, double beta, double x){
		try {
			return Gamma.regularizedGammaQ(alpha, beta / x);
		} catch (MathException e) {
			e.printStackTrace();
		}
		return ILLEGAL_FLAG;
	}	
	
//	public static final double inverse_cumul_dens_function_inv_gamma(double nu, double lambda, double p){
//		try {
//			return nu * lambda / (nu - 2);
//		} catch (MathException e) {
//			e.printStackTrace();
//		}
//		return ILLEGAL_FLAG;
//	}	

	public static double sample_minimum(int[] y) {
		int min = Integer.MAX_VALUE;
		for (int y_i : y){
			if (y_i < min){
				min = y_i;
			}
		}
		return min;
	}

	public static double sample_maximum(int[] y) {
		int max = Integer.MIN_VALUE;
		for (int y_i : y){
			if (y_i > max){
				max = y_i;
			}
		}
		return max;		
	}

	public static double sample_minimum(double[] y){
		double min = Double.MAX_VALUE;
		for (double y_i : y){
			if (y_i < min){
				min = y_i;
			}
		}
		return min;		
	}
	
	public static double sample_maximum(double[] y){
		double max = Double.MIN_VALUE;
		for (double y_i : y){
			if (y_i > max){
				max = y_i;
			}
		}
		return max;			
	}
	/**
	 * Given an array, return the index that houses the maximum value
	 * 
	 * @param arr	the array to be investigated
	 * @return		the index of the greatest value in the array
	 */
	public static int FindMaxIndex(int[] arr){
		int index=0;
		int max=Integer.MIN_VALUE;
		for (int i=0;i<arr.length;i++){
			if (arr[i] > max){
				max=arr[i];
				index=i;
			}				
		}
		return index;
	}

	public static double sample_median(double[] gibbsSamplesForPrediction) {
		int n = gibbsSamplesForPrediction.length;
		Arrays.sort(gibbsSamplesForPrediction);
		if (n % 2 == 0){
			double a = gibbsSamplesForPrediction[n / 2];
			double b = gibbsSamplesForPrediction[n / 2 - 1];
			return (a + b) / 2;
		}
		else {
			return gibbsSamplesForPrediction[(n - 1) / 2];
		}
		
	}

	public static int multinomial_sample(TIntArrayList predictors, double[] weighted_cov_split_prior_subset) {
		double r = StatToolbox.rand();
		double cum_prob = 0;
		int index = 0;
		if (r < weighted_cov_split_prior_subset[0]){
			return predictors.get(0);
		}
		while (true){			
			cum_prob += weighted_cov_split_prior_subset[index];
			if (r > cum_prob && r < cum_prob + weighted_cov_split_prior_subset[index + 1]){
				return predictors.get(index + 1);
			}
			index++;
		}
	}	
}
