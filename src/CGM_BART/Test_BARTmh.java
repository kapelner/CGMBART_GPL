package CGM_BART;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class Test_BARTmh {

	private static CGMBART_gibbs_internal bart;
	private static final int NB = 10;
	private static final int NGAndNB = 20;
	

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
		bart = new CGMBARTRegression();
		bart.setData(Test_CGMBARTTreeNode.data);
		bart.num_gibbs_burn_in = NB;
		bart.num_gibbs_total_iterations = NGAndNB;
		
	}

	@After
	public void tearDown() throws Exception {
	}
	
	
	@Test
	public void testGetErrorsForAllTrees(){
		int num_trees = 10;
		bart.setNumTrees(num_trees);
		bart.InitGibbsSamplingData();
		bart.InitiatizeTrees();

		double[] errors = bart.getErrorsForAllTrees(0);
		
//		System.out.println("Y(" + CGMBART_init.INITIAL_PRED + ") = " + bart.un_transform_y(CGMBART_init.INITIAL_PRED));
//		System.out.println("Y = " + Tools.StringJoin(bart.y, ",") + "  avg_y = " + StatToolbox.sample_average(bart.y));
//		System.out.println("Y_t = " + Tools.StringJoin(bart.y_trans, ",") + "  avg_y_t = " + StatToolbox.sample_average(bart.y_trans));
//		System.out.println("errors = " + Tools.StringJoin(errors, ","));
		
		
		errors = bart.un_transform_y(errors);
		double[] expected_errors = new double[bart.n];
		for (int i = 0; i < bart.n; i++){
			expected_errors[i] = bart.y[i] - num_trees * bart.un_transform_y(CGMBART_init.INITIAL_PRED);
		}
		assertArrayEquals(errors, expected_errors, 0.0001);
		

	}	
	

	
}