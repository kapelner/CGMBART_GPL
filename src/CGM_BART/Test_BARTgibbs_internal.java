package CGM_BART;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;


public class Test_BARTgibbs_internal {

	private static CGMBART_06_gibbs_internal bart;
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
	public void testAssignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(){
		System.out.println("testAssignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq");
		int num_trees = 10;
		bart.setNumTrees(num_trees);
		bart.SetupGibbsSampling();	
		double sigsq = CGMBART_04_init.INITIAL_SIGSQ;
		List<CGMBARTTreeNode> trees = bart.gibbs_samples_of_cgm_trees.get(0);
		for (int t = 0; t < num_trees; t++){
			CGMBARTTreeNode tree = trees.get(t);
			double posterior_sigsq = bart.calcLeafPosteriorVar(tree, sigsq);
			System.out.println("testAssignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq: bart.n = " + bart.n);
			assertEquals(1 / (1 / bart.hyper_sigsq_mu + bart.n / sigsq), posterior_sigsq, 0.0001);
			//draw from posterior distribution
			double posterior_mean = bart.calcLeafPosteriorMean(tree, sigsq, posterior_sigsq);
			bart.assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsqAndUpdateYhats(tree, sigsq);
			double moe = 4 * Math.sqrt(posterior_sigsq);
			System.out.println("testAssignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq\n sigsq = " + sigsq + " sigsq_post = " + posterior_sigsq + " mu_post = " + posterior_mean + " avg_y_node = " + tree.avg_response_untransformed() + " ypred = " + bart.un_transform_y(tree.y_pred));
			assertTrue(tree.y_pred <= posterior_mean + moe && tree.y_pred >= posterior_mean - moe);
			
		}		
	}
	
	@Test
	public void testGetErrorsForAllTrees(){
		int num_trees = 10;
		bart.setNumTrees(num_trees);
		bart.InitGibbsSamplingData();
		bart.InitializeTrees();
		bart.InitializeMus();
		double[] residuals = bart.getResidualsFromFullSumModel(0, new double[bart.n]);
		
//		System.out.println("Y(" + CGMBART_init.INITIAL_PRED + ") = " + bart.un_transform_y(CGMBART_init.INITIAL_PRED));
//		System.out.println("Y = " + Tools.StringJoin(bart.y, ",") + "  avg_y = " + StatToolbox.sample_average(bart.y));
//		System.out.println("Y_t = " + Tools.StringJoin(bart.y_trans, ",") + "  avg_y_t = " + StatToolbox.sample_average(bart.y_trans));
//		System.out.println("errors = " + Tools.StringJoin(errors, ","));
		
		
//		errors = bart.un_transform_y(errors);
		double[] expected_errors = new double[bart.n];
		for (int i = 0; i < bart.n; i++){
			expected_errors[i] = bart.y_trans[i] - num_trees * CGMBART_04_init.INITIAL_PRED;
		}
		assertArrayEquals(expected_errors, residuals, 0.0001);
		

	}
	

	
}