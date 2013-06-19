package CGM_BART;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;

import OpenSourceExtensions.UnorderedPair;

public class Tools {
	
	public static NumberFormat one_digit_format = NumberFormat.getInstance();
	public static NumberFormat two_digit_format = NumberFormat.getInstance();
	static {
		one_digit_format.setMaximumFractionDigits(1);
		two_digit_format.setMaximumFractionDigits(2);
	}
	
	
	@SuppressWarnings("rawtypes")
	public static String StringJoin(ArrayList all, String joinby){
		if (all == null){
			return " NULL ARRAY ";
		}		
		return StringJoin(all.toArray(), joinby);
	}	
	
	public static String StringJoin(TIntArrayList all, String joinby){
		if (all == null){
			return " NULL ARRAY ";
		}		
		return StringJoin(all.toArray(), joinby);
	}	
	/**
	 * Joins a collection of strings into one string
	 * 
	 * @param all		the collection of substrings
	 * @param joinby	the token that joins the substrings
	 * @return			the final product: str1 + joinby + str2 + . . . + strN
	 */	
	public static String StringJoin(double[] all, String joinby){
		if (all == null){
			return " NULL ARRAY ";
		}		
		String joined = "";
		for (int i = 0; i < all.length; i++){
			joined += all[i];
			if (i < all.length - 1)
				joined += joinby;
		}
		return joined;
	}
	public static String StringJoin(int[] all, String joinby){
		if (all == null){
			return " NULL ARRAY ";
		}		
		String joined = "";
		for (int i = 0; i < all.length; i++){
			joined += all[i];
			if (i < all.length - 1)
				joined += joinby;
		}
		return joined;
	}
	public static String StringJoin(TIntArrayList all){
		return StringJoin(all.toArray(), ", ");
	}	
	public static String StringJoin(boolean[] all){
		int[] all_ints = new int[all.length];
		for (int i = 0; i < all.length; i++){
			all_ints[i] = all[i] ? 1 : 0;
		}
		return StringJoin(all_ints, ", ");
	}
	public static String StringJoin(int[] all){
		return StringJoin(all, ", ");
	}
	public static String StringJoin(TDoubleArrayList all){
		return StringJoin(all.toArray(), ", ");
	}		
	public static String StringJoin(double[] all){
		return StringJoin(all, ", ");
	}

	
	public static String StringJoin(ArrayList<Object> all){
		return StringJoin(all, ", ");
	}	
	
	/**
	 * Joins a collection of strings into one string
	 * 
	 * @param all		the collection of substrings
	 * @param joinby	the token that joins the substrings
	 * @return			the final product: str1 + joinby + str2 + . . . + strN
	 */	
	public static String StringJoin(Object[] all, String joinby){
		String joined = "";
		for (int i = 0; i < all.length; i++){
			joined += all[i];
			if (i < all.length - 1)
				joined += joinby;
		}
		return joined;
	}	
	public static String StringJoin(Object[] all){
		return StringJoin(all, ", ");
	}	
	
	/**
	 * Joins a collection of strings into one string
	 * 
	 * @param all		the collection of substrings
	 * @param joinby	the token that joins the substrings
	 * @return			the final product: str1 + joinby + str2 + . . . + strN
	 */	
	public static String StringJoinStrings(Collection<String> all, String joinby){
		Object[] arr = all.toArray();
		String joined = "";
		for (int i = 0; i < arr.length; i++){
			joined += (String)arr[i];
			if (i < arr.length - 1)
				joined += joinby;
		}
		return joined;
	}	
	public static String StringJoin(Collection<String> all){
		return StringJoinStrings(all, ", ");
	}	
	
    public static double max(double[] values) {
    	double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
        	if (value > max){
        		max = value;
        	}
        }
        return max;
    }
    
    public static double min(double[] values) {
    	double min = Double.POSITIVE_INFINITY;
        for (double value : values) {
        	if (value < min){
        		min = value;
        	}
        }
        return min;
    }    
    
    public static double sum_array(double[] arr){
    	double sum = 0;
    	for (int i = 0; i < arr.length; i++){
    		sum += arr[i];
    	}
    	return sum;
    }
    

	public static double sum_inv_array(double[] arr) {
    	double sum = 0;
    	for (int i = 0; i < arr.length; i++){
    		sum += 1 / arr[i];
    	}
    	return sum;
	}	    
 
    public static void weight_arr_by_sum(double[] arr){
    	double weight = sum_array(arr);
    	for (int i = 0; i < arr.length; i++){
    		arr[i] = arr[i] / weight;
    	}
    }
    	
    public static void weight_arr(double[] arr, double weight){
    	for (int i = 0; i < arr.length; i++){
    		arr[i] = arr[i] / weight;
    	}
    }    

	public static double[] subtract_arrays(double[] arr1, double[] arr2) {
		int n = arr1.length;
		double[] diff = new double[n];
		for (int i = 0; i < n; i++){
			diff[i] = arr1[i] - arr2[i];
		}
		return diff;
	}

	public static double[] add_arrays(double[] arr1, double[] arr2) {
		int n = arr1.length;
		double[] sum = new double[n];
		for (int i = 0; i < n; i++){
			sum[i] = arr1[i] + arr2[i];
		}
		return sum;
	}
	
	public static double[] add_arrays(double[] arr1, int[] arr2) {
		int n = arr1.length;
		double[] sum = new double[n];
		for (int i = 0; i < n; i++){
			sum[i] = arr1[i] + arr2[i];
		}
		return sum;
	}	
	
	public static double[] scale_array(double[] arr) {
		double sum = sum_array(arr);
		for (int i = 0; i < arr.length; i++){
			arr[i] = arr[i] /sum;
		}
		return arr;
	}
	
	public static int[] add_arrays(int[] arr1, int[] arr2) {
		int n = arr1.length;
		int[] sum = new int[n];
		for (int i = 0; i < n; i++){
			sum[i] = arr1[i] + arr2[i];
		}
		return sum;
	}	

	public static void print_unordered_pair_set(HashSet<UnorderedPair<Integer>> pairs) {
		for (UnorderedPair<Integer> pair : pairs){
			System.out.print("<" + pair.getFirst() + ", " + pair.getSecond() + ">, ");
		}
		System.out.print("\n");
	}

	public static int[] binary_add_arrays(int[] arr1, int[] arr2) {
		int n = arr1.length;
		int[] sum = new int[n];
		for (int i = 0; i < n; i++){
			sum[i] = (arr1[i] >= 1 ? 1 : 0) + (arr2[i] >= 1 ? 1 : 0);
		}
		return sum;
	}

}
