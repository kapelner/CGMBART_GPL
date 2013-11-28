package CGM_BART;

import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;


public abstract class CGMBART_02_hyperparams extends CGMBART_01_base {

	protected static final double YminAndYmaxHalfDiff = 0.5;
	
	/** all the hyperparameters */
	protected double hyper_mu_mu;
	protected double hyper_sigsq_mu;
	protected double hyper_lambda;
	protected double hyper_k = 2.0;
	protected double hyper_q = 0.9;
	protected double hyper_nu = 3.0;	
	protected double alpha = 0.95;
	protected double beta = 2; //see p271 in CGM10	
	/** information about the response variable */
	protected double y_min;
	protected double y_max;
	protected double y_range_sq;	
	protected Double sample_var_y;
	
	protected static double[] samps_chi_sq_df_eq_nu_plus_n;	
	protected static int samps_chi_sq_df_eq_nu_plus_n_length;
	protected static double[] samps_std_normal;
	protected static int samps_std_normal_length;
	
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		calculateHyperparameters();	
	}
	
	protected void calculateHyperparameters() {
		hyper_mu_mu = 0;
		hyper_sigsq_mu = Math.pow(YminAndYmaxHalfDiff / (hyper_k * Math.sqrt(num_trees)), 2);
		
		if (sample_var_y == null){
			sample_var_y = StatToolbox.sample_variance(y_trans); //0.00001;//
		}

		//calculate lambda from q
		double ten_pctile_chisq_df_hyper_nu = 0;		
		ChiSquaredDistributionImpl chi_sq_dist = new ChiSquaredDistributionImpl(hyper_nu);
		try {
			ten_pctile_chisq_df_hyper_nu = chi_sq_dist.inverseCumulativeProbability(1 - hyper_q);
		} catch (MathException e) {
			System.err.println("Could not calculate inverse cum prob density for chi sq df = " + hyper_nu + " with q = " + hyper_q);
			System.exit(0);
		}

		hyper_lambda = ten_pctile_chisq_df_hyper_nu / hyper_nu * sample_var_y;
}	
	
	public void setK(double hyper_k) {
		this.hyper_k = hyper_k;
	}

	public void setQ(double hyper_q) {
		this.hyper_q = hyper_q;
	}

	public void setNu(double hyper_nu) {
		this.hyper_nu = hyper_nu;
	}
		
	public void setAlpha(double alpha){
		this.alpha = alpha;
	}
	
	public void setBeta(double beta){
		this.beta = beta;
	}		

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
	
	
	public double un_transform_sigsq(double sigsq_t_i){
		//Based on the following elementary calculation: 
		//Var[y^t] = Var[y / R_y] = 1/R_y^2 Var[y]
		return sigsq_t_i * y_range_sq;
	}
	
	public double[] un_transform_sigsq(double[] sigsq_t_is){
		double[] sigsq_is = new double[sigsq_t_is.length];
		for (int i = 0; i < sigsq_t_is.length; i++){
			sigsq_is[i] = un_transform_sigsq(sigsq_t_is[i]);
		}
		return sigsq_is;
	}			
	
	public double un_transform_y_and_round(double yt_i){
		return Double.parseDouble(TreeArrayIllustration.one_digit_format.format((yt_i + YminAndYmaxHalfDiff) * (y_max - y_min) + y_min));
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
