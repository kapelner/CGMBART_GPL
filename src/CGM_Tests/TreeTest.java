package CGM_Tests;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;

import CGM_BART.CGMBART;
import CGM_BART.CGMBARTRegression;
import CGM_BART.CGMBARTTreeNode;

public class TreeTest {

	public static double[] y = {0, 0, 2, 4, 5, 8, 9};
	
	public static ArrayList<double[]> data;
	
	public static CGMBARTTreeNode stump;	
	public static CGMBARTTreeNode simple_tree;
	public static CGMBARTTreeNode double_tree;
	
	
	static {
		data = new ArrayList<double[]>();
		CGMBART bart = new CGMBARTRegression();
		bart.setData(data);
		double[] x_0 = {0, 1, 0, 1, 0, 1, 0};
		double[] x_1 = {15.3, 45.8, 31.2, 9.3, 65.9, 32.3, 9.3};
		double[] x_2 = {1, 1, 1, 1, 0, 0, 0};
		for (int i = 0; i < x_1.length; i++){
			double datum[] = {x_0[i], x_1[i], x_2[i], y[i]};
			data.add(datum);
		}
		
		stump = new CGMBARTTreeNode(null, data, bart);
		
		simple_tree = new CGMBARTTreeNode(null, data, bart);
		simple_tree.splitAttributeM = 0;
		simple_tree.splitValue = 0.0;
		simple_tree.isLeaf = false;
		simple_tree.left = new CGMBARTTreeNode(simple_tree);
		simple_tree.right = new CGMBARTTreeNode(simple_tree);
		CGMBARTTreeNode.propagateRuleChangeOrSwapThroughoutTree(simple_tree, true);	
		
		double_tree = simple_tree.clone(true);
		double_tree.left.isLeaf = false;
		double_tree.left.splitAttributeM = 1;
		double_tree.left.splitValue = 32.3;		
		double_tree.left.left = new CGMBARTTreeNode(double_tree.left);
		double_tree.left.right = new CGMBARTTreeNode(double_tree.left);	
		double_tree.right.isLeaf = false;
		double_tree.right.splitAttributeM = 2;
		double_tree.right.splitValue = 0.0;		
		double_tree.right.left = new CGMBARTTreeNode(double_tree.right);
		double_tree.right.right = new CGMBARTTreeNode(double_tree.right);		
	}

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {

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
	public void testResponses() {
		assertArrayEquals(stump.responses(), y, 0);
	}
	
	@Test
	public void testAvgResponse() {
		assertEquals(stump.avgResponse(), 4, 0.000001);
	}	
	
	@Test
	public void testIsStump() {
		assertTrue(stump.isStump());
	}	

	@Test
	public void testCloneStump() {
		CGMBARTTreeNode cloned_stump = stump.clone();
		assertEquals(cloned_stump.n_at_this_juncture, stump.n_at_this_juncture);
		assertArrayEquals(stump.responses(), cloned_stump.responses(), 0);
		for (int i = 0; i < stump.n_at_this_juncture; i++){
			assertArrayEquals(stump.data.get(i), cloned_stump.data.get(i), 0);
		}
		assertTrue(cloned_stump.isLeaf);
	}	
	
	@Test
	public void testTerminalNodesStump() {
		ArrayList<CGMBARTTreeNode> just_stump = new ArrayList<CGMBARTTreeNode>();
		just_stump.add(stump);
		assertEquals(stump.getTerminalNodes(), just_stump);
		assertEquals(CGMBARTTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 5), just_stump);
		assertEquals(CGMBARTTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 7), just_stump);
		assertTrue(CGMBARTTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 8).size() == 0);
	}
	
	@Test 
	public void testSimpleTreeIntegrity(){
		assertEquals(simple_tree.isStump(), false);
		assertEquals(simple_tree.numLeaves(), 2);
		assertEquals(simple_tree.numPruneNodesAvailable(), 1);
		assertEquals(simple_tree.deepestNode(), 1);
		assertEquals(simple_tree.widestGeneration(), 2);
		assertEquals(simple_tree.splitAttributeM, (Integer)0);
		assertEquals(simple_tree.left.stringLocation(false), "L");
		assertEquals(simple_tree.right.stringLocation(false), "R");		
		double[] left_responses = {0, 2, 5, 9};
		double[] right_responses = {0, 4, 8};
		assertArrayEquals(simple_tree.left.responses(), left_responses, 0);
		assertArrayEquals(simple_tree.right.responses(), right_responses, 0);
		assertEquals(simple_tree.left.sumResponses(), 16, 0);
		assertEquals(simple_tree.right.sumResponses(), 12, 0);
		assertEquals(simple_tree.left.sumResponsesSqd(), 256, 0);
		assertEquals(simple_tree.right.sumResponsesSqd(), 144, 0);	
		ArrayList<CGMBARTTreeNode> internal_nodes = new ArrayList<CGMBARTTreeNode>(1);
		internal_nodes.add(simple_tree);
		assertArrayEquals(CGMBARTTreeNode.findInternalNodes(simple_tree).toArray(), internal_nodes.toArray());
		Object[] just_parent = {simple_tree};
		assertArrayEquals(simple_tree.left.getLineage().toArray(), just_parent);
		assertArrayEquals(simple_tree.right.getLineage().toArray(), just_parent);
	}
	
	@Test 
	public void testDoubleTreeIntegrity(){
		assertEquals(double_tree.isStump(), false);
		assertEquals(double_tree.numLeaves(), 4);
		assertEquals(double_tree.numPruneNodesAvailable(), 2);
		assertEquals(double_tree.deepestNode(), 2);
		assertEquals(double_tree.widestGeneration(), 4);
		assertEquals(double_tree.left.left.stringLocation(false), "LL");
		assertEquals(double_tree.right.right.stringLocation(false), "RR");		
		double[] left_responses = {0, 2, 5, 9};
		double[] right_responses = {0, 4, 8};
		assertArrayEquals(double_tree.left.responses(), left_responses, 0);
		assertArrayEquals(double_tree.right.responses(), right_responses, 0);
		assertEquals(double_tree.left.sumResponses(), 16, 0);
		assertEquals(double_tree.right.sumResponses(), 12, 0);
		assertEquals(double_tree.left.sumResponsesSqd(), 256, 0);
		assertEquals(double_tree.right.sumResponsesSqd(), 144, 0);	
		ArrayList<CGMBARTTreeNode> internal_nodes = new ArrayList<CGMBARTTreeNode>(1);
		internal_nodes.add(double_tree);
		internal_nodes.add(double_tree.left);
		internal_nodes.add(double_tree.right);
		assertArrayEquals(CGMBARTTreeNode.findInternalNodes(double_tree).toArray(), internal_nodes.toArray());
		Object[] left_side = {double_tree.left, double_tree};
		Object[] right_side = {double_tree.right, double_tree};
		assertArrayEquals(double_tree.left.left.getLineage().toArray(), left_side);
		assertArrayEquals(double_tree.left.right.getLineage().toArray(), left_side);
		assertArrayEquals(double_tree.right.left.getLineage().toArray(), right_side);		
		assertArrayEquals(double_tree.right.right.getLineage().toArray(), right_side);	
	}


	@Test 
	public void testSimpleTreePredictorsAndValsAtSplit(){
		//simple tree first
		Integer[] all_predictors = {0, 1, 2};
		assertArrayEquals(simple_tree.predictorsThatCouldBeUsedToSplitAtNode().toArray(), all_predictors);
		assertEquals(simple_tree.pAdj(), 3);
		Object[] vals_to_split = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
		assertArrayEquals(simple_tree.possibleSplitValuesGivenAttribute().toArray(), vals_to_split);
		assertEquals(simple_tree.pAdj(), 3);
		//now go into the leaves and see what else we can split on
		Integer[] predictors_left = {1, 2};
		assertArrayEquals(simple_tree.left.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left);
		assertArrayEquals(simple_tree.right.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left);

		
	}
	
	@Test 
	public void testDoubleTreePredictorsAtSplit(){
		Integer[] all_predictors = {0, 1, 2};
		assertArrayEquals(double_tree.predictorsThatCouldBeUsedToSplitAtNode().toArray(), all_predictors);
		Integer[] predictors_left_on_left = {1, 2};
		assertArrayEquals(double_tree.left.left.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left_on_left);
		Integer[] predictors_left_on_right = {1};
		assertArrayEquals(double_tree.right.right.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left_on_right);
		
		//now take it a step further... extend the tree on the left left and see what happens
		CGMBARTTreeNode double_tree_ext = buildDoubleTreeExt();
		assertArrayEquals(double_tree_ext.left.left.left.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left_on_left);
		
	}	
	
	private CGMBARTTreeNode buildDoubleTreeExt(){
		CGMBARTTreeNode double_tree_ext = double_tree.clone(true);
		double_tree_ext.left.left.isLeaf = false;
		double_tree_ext.left.left.splitAttributeM = 1;
		double_tree_ext.left.left.splitValue = 31.2;
		double_tree_ext.left.left.left = new CGMBARTTreeNode(double_tree_ext.left.left);
		double_tree_ext.left.left.left.splitAttributeM = 1;
		//now we want to make sure it has the same num predictors
		return double_tree_ext;				
	}
	
	@Test 
	public void testDoubleTreeNadj(){
		CGMBARTTreeNode double_tree_ext = buildDoubleTreeExt();
		
		assertEquals(double_tree_ext.nAdj(), 7);
		assertEquals(double_tree_ext.right.nAdj(), 7);
		assertEquals(double_tree_ext.left.nAdj(), 7);
		assertEquals(double_tree_ext.left.left.nAdj(), 4);
	}	

	
}
