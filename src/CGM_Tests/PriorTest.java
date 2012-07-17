package CGM_Tests;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import CGM_BART.CGMBARTPriorBuilder;
import CGM_Statistics.CGMTreeNode;

public class PriorTest {

	public static CGMBARTPriorBuilder prior_builder;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		prior_builder = new CGMBARTPriorBuilder(TreeTest.data);
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test 
	public void testSimpleTreePriorBuilderConstructor(){
		assertEquals(prior_builder.getP(), 3);
		double[] min_vals = {0, 9.3, 0};
		assertArrayEquals(prior_builder.getMinimumValuesByAttribute(), min_vals, 0);
	}

	@Test 
	public void testSimpleTreePriorBuilderPredictorsAndValsAtSplit(){
		//simple tree first
		Integer[] all_predictors = {0, 1, 2};
		TreeTest.simple_tree.predictors_that_can_be_assigned = prior_builder.predictorsThatCouldBeUsedToSplitAtNode(TreeTest.simple_tree);
		assertArrayEquals(TreeTest.simple_tree.predictors_that_can_be_assigned.toArray(), all_predictors);
		assertEquals(TreeTest.simple_tree.pAdj(), 3);
		Object[] vals_to_split = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
		assertArrayEquals(prior_builder.possibleSplitValues(TreeTest.simple_tree).toArray(), vals_to_split);
		assertEquals(TreeTest.simple_tree.pAdj(), 3);
		//now go into the leaves and see what else we can split on
		Integer[] predictors_left = {1, 2};
		assertArrayEquals(prior_builder.predictorsThatCouldBeUsedToSplitAtNode(TreeTest.simple_tree.left).toArray(), predictors_left);
		assertArrayEquals(prior_builder.predictorsThatCouldBeUsedToSplitAtNode(TreeTest.simple_tree.right).toArray(), predictors_left);

		
	}
	
	@Test 
	public void testDoubleTreePriorBuilderPredictorsAtSplit(){
		Integer[] all_predictors = {0, 1, 2};
		assertArrayEquals(prior_builder.predictorsThatCouldBeUsedToSplitAtNode(TreeTest.double_tree).toArray(), all_predictors);
		Integer[] predictors_left_on_left = {1, 2};
		assertArrayEquals(prior_builder.predictorsThatCouldBeUsedToSplitAtNode(TreeTest.double_tree.left.left).toArray(), predictors_left_on_left);
		Integer[] predictors_left_on_right = {1};
		assertArrayEquals(prior_builder.predictorsThatCouldBeUsedToSplitAtNode(TreeTest.double_tree.right.right).toArray(), predictors_left_on_right);
		
		//now take it a step further... extend the tree on the left left and see what happens
		CGMTreeNode double_tree_ext = TreeTest.double_tree.clone(true);
		double_tree_ext.left.left.isLeaf = false;
		double_tree_ext.left.left.left = new CGMTreeNode(double_tree_ext.left.left, null);
		double_tree_ext.left.left.left.splitAttributeM = 1;
		//now we want to make sure it has the same num predictors
		double_tree_ext.left.left.left.predictors_that_can_be_assigned = prior_builder.predictorsThatCouldBeUsedToSplitAtNode(double_tree_ext.left.left.left);
		assertArrayEquals(double_tree_ext.left.left.left.predictors_that_can_be_assigned.toArray(), predictors_left_on_right);
		
	}	
}
