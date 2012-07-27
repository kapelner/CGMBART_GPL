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
	
	@Test //y = {0, 0, 2, 4, 5, 8, 9}; avg = 4
	public void testGetResidualsBySubtractingTrees(){
		int num_trees = 10;
		bart.setNumTrees(num_trees);
		bart.SetupGibbsSampling();
		List<CGMBARTTreeNode> old_trees_all_but_one = bart.gibbs_samples_of_cgm_trees.get(0).subList(0, num_trees - 1);
		for (CGMBARTTreeNode old_tree : old_trees_all_but_one){
			old_tree.y_prediction = 0.0;
		}
		double[] resids = bart.getResidualsBySubtractingTrees(old_trees_all_but_one);
		resids = bart.un_transform_y(resids);
		System.out.println(Tools.StringJoin(resids, ","));
	}
	
}