package CGM_BART;

import java.util.ArrayList;
import java.util.Collection;

public class Tools {
	/**
	 * Joins a collection of strings into one string
	 * 
	 * @param all		the collection of substrings
	 * @param joinby	the token that joins the substrings
	 * @return			the final product: str1 + joinby + str2 + . . . + strN
	 */	
	public static String StringJoin(double[] all, String joinby){
		String joined = "";
		for (int i = 0; i < all.length; i++){
			joined += all[i];
			if (i < all.length - 1)
				joined += joinby;
		}
		return joined;
	}
	public static String StringJoin(double[] all){
		return StringJoin(all, ", ");
	}
	
	public static String StringJoin(ArrayList<Double> all, String joinby){
		return StringJoin(all.toArray(), joinby);
	}	
	public static String StringJoin(ArrayList<Double> all){
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
	public static String StringJoin(Collection<String> all, String joinby){
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
		return StringJoin(all, ", ");
	}	
}
