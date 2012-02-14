package CGM_Matrices;

import java.util.LinkedHashMap;

import GemIdentTools.Matrices.DoubleMatrix;

/**
 * This class just extends DoubleMatrix by caching some data that's generated many times
 * 
 * @author kapelner
 *
 */
public class CGMDoubleMatrix extends DoubleMatrix {
	private static final long serialVersionUID = 3305694930946605883L;	
	
	private static final int NUM_IN_CACHE = 400;
	private static final LinkedHashMap<Integer, DoubleMatrix> JnOverNs;
//	private static final LinkedHashMap<Integer, DoubleMatrix> INs;
	static {
		JnOverNs = new LinkedHashMap<Integer, DoubleMatrix>();
//		INs = new LinkedHashMap<Integer, DoubleMatrix>();
	}
	
	private static DoubleMatrix MakeJnOverN(int n){
		DoubleMatrix jn_over_n = new DoubleMatrix(n, n);
		double val = 1 / (double)n;
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++){
				jn_over_n.set(i, j, val);
			}			
		}
		return jn_over_n;
	}
	
    public static DoubleMatrix CachedJnOverN(int n){
    	DoubleMatrix jn_over_n = JnOverNs.get(n);
    	if (jn_over_n == null){
    		//generate it and cache it
    		jn_over_n = MakeJnOverN(n);
    		cachejn(n, jn_over_n);
    	}
    	else {
//    		System.out.println("loaded Jnovern for n = " + n);
    	}
	 	return jn_over_n;
    }  

    private static void cachejn(int n, DoubleMatrix jn_over_n) {
    	if (JnOverNs.size() == NUM_IN_CACHE){
//    		System.err.println("kicking one out of JnOverNs");
    		for (int key_n : JnOverNs.keySet()){ //Java is limited...
    			JnOverNs.remove(key_n); //remove first element then bounce
    			break;
    		}
    	}
//    	System.err.println("caching Jnovern for n = " + n);
    	JnOverNs.put(n, jn_over_n);
	}

//	public static DoubleMatrix CachedIn(int n){
//    	DoubleMatrix in = INs.get(n);
//    	if (in == null){
//    		//generate it and cache it
//    		in = DoubleMatrix.In(n);
//    		cachein(n, in);    		
//    	}
//	 	return in; 	   
//    }
//	
//	private static void cachein(int n, DoubleMatrix in) {
//    	if (INs.size() == NUM_IN_CACHE){ 
////    		System.out.println("kicking one out of INs");
//    		for (int key_n : INs.keySet()){ //Java is limited...
//    			INs.remove(key_n); //remove first element then bounce
//    			break;
//    		}
//    	}	
////    	System.out.println("caching In for n = " + n);
//		INs.put(n, in);
//	}

	public CGMDoubleMatrix(int m, int n) {
		super(m, n);
	}

	
}
