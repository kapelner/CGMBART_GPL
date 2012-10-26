package CGM_BART;

import gnu.trove.list.array.TDoubleArrayList;

import java.io.Serializable;
import java.util.ArrayList;


public abstract class CGMBART_02_hyperparams extends CGMBART_01_base implements Serializable {
	private static final long serialVersionUID = -7460897154338844402L;
	
	protected static final double YminAndYmaxHalfDiff = 0.5;
	
	/** all the hyperparameters */
	protected double hyper_mu_mu;
	protected double hyper_sigsq_mu;
	protected double hyper_nu;
	protected double hyper_lambda;
	/** information about the response variable */
	protected double y_min;
	protected double y_max;
	protected double y_range_sq;	
	
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		calculateHyperparameters();	
	}		
	
	// hist(1 / rgamma(5000, 1.5, 1.5 * 153.65), br=100)
	protected void calculateHyperparameters() {
//		System.out.println("calculateHyperparameters in BART\n\n");
		double k = 2; //StatToolbox.inv_norm_dist(1 - (1 - CGMShared.MostOfTheDistribution) / 2.0);	
//		y_min = StatToolbox.sample_minimum(y);
//		y_max = StatToolbox.sample_maximum(y);
		hyper_mu_mu = 0;
		hyper_sigsq_mu = Math.pow(YminAndYmaxHalfDiff / (k * Math.sqrt(num_trees)), 2);
		
		//first calculate \sigma_\mu
		
		//we fix nu and q		
		hyper_nu = 3.0;
		
		//now we do a simple search for the best value of lambda
		//if sig_sq ~ \nu\lambda * X where X is Inv chi sq, then sigsq ~ InvGamma(\nu/2, \nu\lambda/2) \neq InvChisq
		double s_sq_y = StatToolbox.sample_variance(y_trans); //0.00001;//
//		double prob_diff = Double.MAX_VALUE;
		
//		double q = 0.9;
		double ten_pctile_chisq_df_3 = 0.5843744; //we need q=0.9 for this to work
		
		hyper_lambda = ten_pctile_chisq_df_3 / 3 * s_sq_y;
//		System.out.println("lambda: " + lambda);
//		
//		for (lambda = 0.00001; lambda < 10 * s_sq_y; lambda += (s_sq_y / 10000)){
//			double p = StatToolbox.cumul_dens_function_inv_gamma(hyper_nu / 2, hyper_nu * lambda / 2, s_sq_y);			
//			if (Math.abs(p - q) < prob_diff){
////				System.out.println("hyper_lambda = " + hyper_lambda + " lambda = " + lambda + " p = " + p + " ssq = " + ssq);
//				hyper_lambda = lambda;
//				prob_diff = Math.abs(p - q);
//			}
//		}
		System.out.println("y_min = " + y_min + " y_max = " + y_max + " R_y = " + Math.sqrt(y_range_sq));
		System.out.println("hyperparams:  k = " + k + " hyper_mu_mu = " + hyper_mu_mu + " sigsq_mu = " + hyper_sigsq_mu + " hyper_lambda = " + hyper_lambda + " hyper_nu = " + hyper_nu + " s_y_trans^2 = " + s_sq_y + " R_y = " + Math.sqrt(y_range_sq) + "\n\n");
	}	
	
	
	//make sure you get the prior correct if you don't transform
	
	protected void transformResponseVariable() {
//		System.out.println("CGMBART transformResponseVariable");
		//make sure to initialize the y_trans to be y first
		super.transformResponseVariable();
		//make data we need later
		y_min = StatToolbox.sample_minimum(y_orig);
		y_max = StatToolbox.sample_maximum(y_orig);
		y_range_sq = Math.pow(y_max - y_min, 2);
	
		for (int i = 0; i < n; i++){
			y_trans[i] = transform_y(y_orig[i]);
		}
		//debug stuff
	//	y_and_y_trans.println("y,y_trans");
	//	for (int i = 0; i < n; i++){
	//		System.out.println("y_trans[i] = " + y_trans[i] + " y[i] = " + y[i] + " y_untransform = " + un_transform_y(y_trans[i]));
	//		y_and_y_trans.println(y[i] + "," + y_trans[i]);
	//	}
	//	y_and_y_trans.close();
	}

	public double transform_y(double y_i){
		return (y_i - y_min) / (y_max - y_min) - YminAndYmaxHalfDiff;
	}
	
	public double[] un_transform_y(double[] yt){
		double[] y = new double[yt.length];
		for (int i = 0; i < yt.length; i++){
			y[i] = un_transform_y(yt[i]);
		}
		return y;
	}
	
	public double un_transform_y(double yt_i){
		return (yt_i + YminAndYmaxHalfDiff) * (y_max - y_min) + y_min;
	}
	
	public double un_transform_y(Double yt_i){
		if (yt_i == null){
			return -9999999;
		}
		return un_transform_y((double)yt_i);
	}	
	
	public double un_transform_y_and_round(double yt_i){
		double y_trans_i = (yt_i + YminAndYmaxHalfDiff) * (y_max - y_min) + y_min;
		return Double.parseDouble(TreeIllustration.two_digit_format.format(y_trans_i));
	}
	
	public double[] un_transform_y_and_round(double[] yt){
		double[] y = new double[yt.length];
		for (int i = 0; i < yt.length; i++){
			y[i] = un_transform_y_and_round(yt[i]);
		}
		return y;
	}		

	public double[] un_transform_y_and_round(TDoubleArrayList yt){
		return un_transform_y_and_round(yt.toArray());
	}
	
	public double getHyper_mu_mu() {
		return hyper_mu_mu;
	}

	public double getHyper_sigsq_mu() {
		return hyper_sigsq_mu;
	}

	public double getHyper_nu() {
		return hyper_nu;
	}

	public double getHyper_lambda() {
		return hyper_lambda;
	}

	public double getY_min() {
		return y_min;
	}

	public double getY_max() {
		return y_max;
	}

	public double getY_range_sq() {
		return y_range_sq;
	}
}
