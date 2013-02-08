package CGM_BART;

import gnu.trove.list.array.TDoubleArrayList;

import java.io.Serializable;
import java.util.ArrayList;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;


public abstract class CGMBART_02_hyperparams extends CGMBART_01_base implements Serializable {
	private static final long serialVersionUID = -7460897154338844402L;
	
	protected static final double YminAndYmaxHalfDiff = 0.5;
	
	protected static double hyper_k = 2.0; //StatToolbox.inv_norm_dist(1 - (1 - CGMShared.MostOfTheDistribution) / 2.0);	
	protected static double hyper_q = 0.9;
	protected static double hyper_nu = 3.0;
	
	/** all the hyperparameters */
	protected double hyper_mu_mu;
	protected double hyper_sigsq_mu;
	protected double hyper_lambda;
	/** information about the response variable */
	protected double y_min;
	protected double y_max;
	protected double y_range_sq;	
	protected Double sample_var_y;
	
	protected transient double[] samps_chi_sq_df_eq_nu_plus_n;	
	
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		calculateHyperparameters();	
//		for (int i = 0; i < n; i++){
//			System.out.println("i: " + i + " ROW: " + Tools.StringJoin(X_y.get(i), "\t"));
//		}
		StatToolbox.cacheInvGammas(n, hyper_nu, this);
	}		
	
	// hist(1 / rgamma(5000, 1.5, 1.5 * 153.65), br=100)
	protected void calculateHyperparameters() {
//		System.out.println("calculateHyperparameters in BART\n\n");
		hyper_mu_mu = 0;
		hyper_sigsq_mu = Math.pow(YminAndYmaxHalfDiff / (hyper_k * Math.sqrt(num_trees)), 2);
//		System.out.println("hyper_sigsq_mu: " + hyper_sigsq_mu);
		
		//now we do a simple search for the best value of lambda
		//if sig_sq ~ \nu\lambda * X where X is Inv chi sq, then sigsq ~ InvGamma(\nu/2, \nu\lambda/2) \neq InvChisq
//		System.out.println("y_trans: " + Tools.StringJoin(y_trans));
		
		if (sample_var_y == null){
			sample_var_y = StatToolbox.sample_variance(y_trans); //0.00001;//
		}

		//calculate lambda from q
		double ten_pctile_chisq_df_hyper_nu = 0;		
		ChiSquaredDistributionImpl chi_sq_dist = new ChiSquaredDistributionImpl(hyper_nu);
		try {
			ten_pctile_chisq_df_hyper_nu = chi_sq_dist.inverseCumulativeProbability(1 - hyper_q);
		} catch (MathException e) {
			System.err.println("Could not calculate inverse cum prob density for chi sq df = " + hyper_nu);
			System.exit(0);
		}

		hyper_lambda = ten_pctile_chisq_df_hyper_nu / hyper_nu * sample_var_y;
//		System.out.println("hyper_lambda via invcumprob: " + hyper_lambda);			

//		OLD GRID METHOD BEFORE 
//		double prob_diff = Double.MAX_VALUE;
//		for (double lambda = 0.00001; lambda < 10 * sample_var_y; lambda += (sample_var_y / 100000)){
//			double p = StatToolbox.cumul_dens_function_inv_gamma(hyper_nu / 2, hyper_nu * lambda / 2, sample_var_y);			
//			if (Math.abs(p - hyper_q) < prob_diff){
////				System.out.println("hyper_lambda = " + hyper_lambda + " lambda = " + lambda + " p = " + p + " ssq = " + ssq);
//				hyper_lambda = lambda;
//				prob_diff = Math.abs(p - hyper_q);
//			}
//		}
//		System.out.println("hyper_lambda via grid: " + hyper_lambda);
//		System.out.println("y_min = " + y_min + " y_max = " + y_max + " R_y = " + Math.sqrt(y_range_sq));
//		System.out.println("hyperparams:  k = " + hyper_k + " hyper_mu_mu = " + hyper_mu_mu + " sigsq_mu = " + hyper_sigsq_mu + " hyper_lambda = " + hyper_lambda + " hyper_nu = " + hyper_nu + " hyper_q = " + hyper_q + " s_y_trans^2 = " + sample_var_y + " R_y = " + Math.sqrt(y_range_sq) + "\n\n");
	}	
	
	public static void setK(double hyper_k) {
		CGMBART_02_hyperparams.hyper_k = hyper_k;
	}

	public static void setQ(double hyper_q) {
		CGMBART_02_hyperparams.hyper_q = hyper_q;
	}

	public static void setNU(double hyper_nu) {
		CGMBART_02_hyperparams.hyper_nu = hyper_nu;
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
	
	//Var[y^t] = Var[y / R_y] = 1/R_y^2 Var[y]
	public double un_transform_sigsq(double sigsq_t_i){
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
