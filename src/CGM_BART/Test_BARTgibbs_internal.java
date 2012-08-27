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
	public void testFindOtherTrees(){
		int num_trees = 10;
		bart.setNumTrees(num_trees);
		bart.SetupGibbsSampling();
		ArrayList<CGMBARTTreeNode> old_trees = bart.gibbs_samples_of_cgm_trees.get(0);
		bart.DoOneGibbsSampleAndIncrement();
		ArrayList<CGMBARTTreeNode> new_trees = bart.gibbs_samples_of_cgm_trees.get(1);
		//if we find all other trees on the zeroth go, we should get back the old trees without the first
		List<CGMBARTTreeNode> expected_trees = null;
		expected_trees = old_trees.subList(1, num_trees);
		assertArrayEquals(bart.findOtherTrees(1, 0).toArray(), expected_trees.toArray());
		//so now we take the second tree. So we need a new first tree and then the rest old
		expected_trees = new_trees.subList(0, 1);
		expected_trees.addAll(old_trees.subList(2, num_trees));
		assertArrayEquals(bart.findOtherTrees(1, 1).toArray(), expected_trees.toArray());		
		//so now we take the fifth tree. So we need a new first four trees and then the rest old
		expected_trees = new_trees.subList(0, 4);
		expected_trees.addAll(old_trees.subList(5, num_trees));
		assertArrayEquals(bart.findOtherTrees(1, 4).toArray(), expected_trees.toArray());	
		//now we do the last tree
		expected_trees = new_trees.subList(0, num_trees - 1);
		assertArrayEquals(bart.findOtherTrees(1, num_trees - 1).toArray(), expected_trees.toArray());			
	}
	
	@Test //y = {0, 0, 2, 4, 5, 8, 9}; avg = 4.0
	public void testGetResidualsBySubtractingTrees(){
		int num_trees = 10;
		double y_pred_initial = -0.4;
		bart.setNumTrees(num_trees);
		bart.SetupGibbsSampling();
		List<CGMBARTTreeNode> old_trees_all_but_one = bart.gibbs_samples_of_cgm_trees.get(0).subList(0, num_trees - 1);
		for (CGMBARTTreeNode old_tree : old_trees_all_but_one){
			old_tree.y_prediction = y_pred_initial;
		}
		double[] resids = bart.getResidualsBySubtractingTrees(old_trees_all_but_one);
//		resids = bart.un_transform_y(resids);
		System.out.println("Y = " + Tools.StringJoin(bart.y, ",") + "  avg_y = " + StatToolbox.sample_average(bart.y));
		System.out.println("Y_t = " + Tools.StringJoin(bart.y_trans, ",") + "  avg_y_t = " + StatToolbox.sample_average(bart.y_trans));
		System.out.println("rjs = " + Tools.StringJoin(resids, ","));
		double[] expected_resids = new double[bart.n];
		for (int i = 0; i < bart.n; i++){
			expected_resids[i] = bart.y[i] - (num_trees - 1) * y_pred_initial;
		}
		assertArrayEquals(resids, expected_resids, 0.0001);
	}
	
	@Test
	public void testAssignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(){
		int num_trees = 10;
		bart.setNumTrees(num_trees);
		bart.SetupGibbsSampling();	
		double sigsq = bart.gibbs_samples_of_sigsq.get(0);
		List<CGMBARTTreeNode> trees = bart.gibbs_samples_of_cgm_trees.get(0);
		for (int t = 0; t < num_trees; t++){
			CGMBARTTreeNode tree = trees.get(t);
			double posterior_sigsq = bart.calcLeafPosteriorVar(tree, sigsq);
			//draw from posterior distribution
			double posterior_mean = bart.calcLeafPosteriorMean(tree, sigsq, posterior_sigsq);			
			bart.assignLeafValsBySamplingFromPosteriorMeanGivenCurrentSigsq(tree, sigsq);
			System.out.println("sigsq = " + sigsq + " sigsq_post = " + posterior_sigsq + " mu_post = " + posterior_mean + " avg_y_node = " + tree.avg_response_untransformed() + " ypred = " + bart.un_transform_y(tree.y_prediction));
		}		
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
		
		
//		errors = bart.un_transform_y(errors);
		double[] expected_errors = new double[bart.n];
		for (int i = 0; i < bart.n; i++){
			expected_errors[i] = bart.y[i] - num_trees * CGMBART_04_init.INITIAL_PRED;
		}
		assertArrayEquals(errors, expected_errors, 0.0001);
		

	}
	

	
}