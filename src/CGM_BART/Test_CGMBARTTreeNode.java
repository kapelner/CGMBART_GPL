package CGM_BART;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;


public class Test_CGMBARTTreeNode {

	
	
	public static ArrayList<double[]> data;
	
	public static CGMBARTTreeNode stump;	
	public static CGMBARTTreeNode simple_tree;
	public static CGMBARTTreeNode double_tree;

	public static CGMBARTRegression bart;
	
	public static double[] y = {0, 0, 2, 4, 5, 8, 9};
	static {
		data = new ArrayList<double[]>();
		bart = new CGMBARTRegression();
		
		double[] x_0 = {0, 1, 0, 1, 0, 1, 0};
		double[] x_1 = {15.3, 45.8, 31.2, 9.3, 65.9, 32.3, 9.3};
		double[] x_2 = {1, 1, 1, 1, 0, 0, 0};
		for (int i = 0; i < x_1.length; i++){
			double datum[] = {x_0[i], x_1[i], x_2[i], y[i]};
			data.add(datum);
		}
		bart.setData(data);
		data = bart.getData();
		
		stump = new CGMBARTTreeNode(bart);
		stump.n_eta = data.size();
		
		simple_tree = new CGMBARTTreeNode(bart);
		simple_tree.n_eta = data.size();
		
		simple_tree.splitAttributeM = 0;
		simple_tree.splitValue = 0.0;
		simple_tree.isLeaf = false;
		simple_tree.left = new CGMBARTTreeNode(simple_tree);
		simple_tree.right = new CGMBARTTreeNode(simple_tree);
		simple_tree.propagateDataByChangedRule();
		
		double_tree = simple_tree.clone();
		double_tree.left.isLeaf = false;
		double_tree.left.splitAttributeM = 1;
		double_tree.left.splitValue = 31.2;		
		double_tree.left.left = new CGMBARTTreeNode(double_tree.left);
		double_tree.left.right = new CGMBARTTreeNode(double_tree.left);	
		double_tree.right.isLeaf = false;
		double_tree.right.splitAttributeM = 2;
		double_tree.right.splitValue = 0.0;		
		double_tree.right.left = new CGMBARTTreeNode(double_tree.right);
		double_tree.right.right = new CGMBARTTreeNode(double_tree.right);
		
		double_tree.propagateDataByChangedRule();
		System.out.println("create double_tree n_eta's\nP: " + 
				double_tree.n_eta + " L: " + double_tree.left.n_eta + " R: " + double_tree.right.n_eta + 
				" LL: " + double_tree.left.left.n_eta + " LR: " + double_tree.left.right.n_eta + 
				" RL: " + double_tree.right.left.n_eta + " RR: " + double_tree.right.right.n_eta);
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
		System.out.println("y: " + Tools.StringJoin(y));
		System.out.println("stump.getResponses(): " + Tools.StringJoin(stump.responses));
		Arrays.sort(stump.responses);
		assertArrayEquals(y, stump.responses, 0);
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
		assertEquals(cloned_stump.n_eta, stump.n_eta);
		double[] stump_responses = stump.responses.clone();
		double[] cloned_stump_responses = cloned_stump.responses.clone();
		Arrays.sort(stump_responses);
		Arrays.sort(cloned_stump_responses);
		assertArrayEquals(stump_responses, cloned_stump_responses, 0);
		assertTrue(cloned_stump.isLeaf);
	}	
	
	@Test
	public void testTerminalNodesStump() {
		ArrayList<CGMBARTTreeNode> just_stump = new ArrayList<CGMBARTTreeNode>();
		just_stump.add(stump);
		assertEquals(stump.getTerminalNodes(), just_stump);
		assertEquals(CGMBARTTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 5), just_stump);
		assertEquals(CGMBARTTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 7), just_stump);
		System.out.println("testTerminalNodesStump:  " + CGMBARTTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 8).size());
		assertTrue(CGMBARTTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 8).size() == 0);
	}
	
	@Test 
	public void testSimpleTreeIntegrity(){
		assertEquals(simple_tree.isStump(), false);
		assertEquals(simple_tree.numLeaves(), 2);
		assertEquals(simple_tree.numPruneNodesAvailable(), 1);
		assertEquals(simple_tree.deepestNode(), 1);
//		assertEquals(simple_tree.widestGeneration(), 2);
		assertEquals(simple_tree.splitAttributeM, 0);
		assertEquals(simple_tree.left.stringLocation(false), "L");
		assertEquals(simple_tree.right.stringLocation(false), "R");		
		double[] left_responses = {0, 2, 5, 9};
		double[] right_responses = {0, 4, 8};
		assertArrayEquals(simple_tree.left.responses, left_responses, 0);
		assertArrayEquals(simple_tree.right.responses, right_responses, 0);
		assertEquals(simple_tree.right.sumResponses(), 12, 0);
		assertEquals(simple_tree.left.sumResponses(), 16, 0);		
		assertEquals(simple_tree.left.sumResponsesQuantitySqd(), 256, 0);
		assertEquals(simple_tree.right.sumResponsesQuantitySqd(), 144, 0);	
		ArrayList<CGMBARTTreeNode> internal_nodes = new ArrayList<CGMBARTTreeNode>(1);
		internal_nodes.add(simple_tree);
//		assertArrayEquals(CGMBARTTreeNode.findInternalNodes(simple_tree).toArray(), internal_nodes.toArray());
		Object[] just_parent = {simple_tree};
		assertArrayEquals(simple_tree.left.getLineage().keySet().toArray(), just_parent);
		assertArrayEquals(simple_tree.right.getLineage().keySet().toArray(), just_parent);
	}
	
	@Test 
	public void testDoubleTreeIntegrity(){
		assertEquals(double_tree.isStump(), false);
		assertEquals(double_tree.numLeaves(), 4);
		assertEquals(double_tree.numPruneNodesAvailable(), 2);
		assertEquals(double_tree.deepestNode(), 2);
//		assertEquals(double_tree.widestGeneration(), 4);
		assertEquals(double_tree.left.left.stringLocation(false), "LL");
		assertEquals(double_tree.right.right.stringLocation(false), "RR");		

		HashSet<Double> left_responses = new HashSet<Double>();
		left_responses.add(9.0);
		left_responses.add(0.0);
		left_responses.add(2.0);
		left_responses.add(5.0);
		HashSet<Double> right_responses = new HashSet<Double>();
		right_responses.add(8.0);
		right_responses.add(0.0);
		right_responses.add(4.0);

		assertEquals(double_tree.left.sumResponses(), 16, 0);
		assertEquals(double_tree.right.sumResponses(), 12, 0);
		assertEquals(double_tree.left.sumResponsesQuantitySqd(), 256, 0);
		assertEquals(double_tree.right.sumResponsesQuantitySqd(), 144, 0);	
		ArrayList<CGMBARTTreeNode> internal_nodes = new ArrayList<CGMBARTTreeNode>(1);
		internal_nodes.add(double_tree);
		internal_nodes.add(double_tree.left);
		internal_nodes.add(double_tree.right);
//		assertArrayEquals(CGMBARTTreeNode.findInternalNodes(double_tree).toArray(), internal_nodes.toArray());
		Object[] left_side = {double_tree.left, double_tree};
		Object[] right_side = {double_tree.right, double_tree};
		assertArrayEquals(double_tree.left.left.getLineage().keySet().toArray(), left_side);
		assertArrayEquals(double_tree.left.right.getLineage().keySet().toArray(), left_side);
		assertArrayEquals(double_tree.right.left.getLineage().keySet().toArray(), right_side);		
		assertArrayEquals(double_tree.right.right.getLineage().keySet().toArray(), right_side);	
	}


	@Test 
	public void testSimpleTreePredictorsAndValsAtSplit(){
		//simple tree first
		int[] all_predictors = {0, 1, 2};
		assertArrayEquals(simple_tree.predictorsThatCouldBeUsedToSplitAtNode().toArray(), all_predictors);
		assertEquals(simple_tree.pAdj(), 3);
		double[] vals_to_split = {0.0}; //remember we can't split on 1 because it's the max value
		System.out.println("poss splits: " + Tools.StringJoin(simple_tree.possibleSplitValuesGivenAttribute().toArray(), ","));
		double[] arr1 = simple_tree.possibleSplitValuesGivenAttribute().toArray();
		for (int i = 0; i < arr1.length; i++){
			assertEquals(arr1[i], vals_to_split[i], 0.0001);
		}
		assertEquals(simple_tree.pAdj(), 3);
		//now go into the leaves and see what else we can split on
		int[] predictors_left = {1, 2};
		assertArrayEquals(simple_tree.left.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left);
		assertArrayEquals(simple_tree.right.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left);

		
	}
	
	@Test 
	public void testDoubleTreePredictorsAtSplit(){
		int[] all_predictors = {0, 1, 2};
		assertArrayEquals(double_tree.predictorsThatCouldBeUsedToSplitAtNode().toArray(), all_predictors);
		assertEquals(3, double_tree.pAdj());
		assertEquals(1, double_tree.nAdj());
		
		//Test double.tree n.adj at second nodes
		assertEquals(3, double_tree.left.nAdj());
		assertEquals(1, double_tree.right.nAdj());	
		
		int[] predictors_left_left = {1, 2};
		assertArrayEquals(double_tree.left.left.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left_left);
		assertEquals(double_tree.left.left.pAdj(), 2);
		
		int[] predictors_left_right = {};
		assertArrayEquals(double_tree.left.right.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left_right);
		assertEquals(0, double_tree.left.right.pAdj());
		
		int[] predictors_right_right = {1};
		assertArrayEquals(double_tree.right.right.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_right_right);
		assertEquals(1, double_tree.right.right.pAdj());
		
		int[] predictors_right_left = {};
		assertArrayEquals(double_tree.right.left.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_right_left);
		assertEquals(0, double_tree.right.left.pAdj());
		
		//now take it a step further... extend the tree on the left left and see what happens
		CGMBARTTreeNode double_tree_ext = buildDoubleTreeExt();
		
		int[] predictors_left_left_left = {1, 2};
		assertArrayEquals(double_tree_ext.left.left.left.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left_left_left);
		assertEquals(double_tree_ext.left.left.left.pAdj(), 2);
		
		int[] predictors_left_left_right = {};
		assertArrayEquals(double_tree_ext.left.left.right.predictorsThatCouldBeUsedToSplitAtNode().toArray(), predictors_left_left_right);
		assertEquals(double_tree_ext.left.left.right.pAdj(), 0);		
	}	
	
	private CGMBARTTreeNode buildDoubleTreeExt(){
		CGMBARTTreeNode double_tree_ext = double_tree.clone();
		double_tree_ext.left.left.isLeaf = false;
		double_tree_ext.left.left.splitAttributeM = 1;
		double_tree_ext.left.left.splitValue = 15.3;
		double_tree_ext.left.left.left = new CGMBARTTreeNode(double_tree_ext.left.left);
		double_tree_ext.left.left.right = new CGMBARTTreeNode(double_tree_ext.left.left);
		double_tree_ext.propagateDataByChangedRule();
		//now we want to make sure it has the same num predictors
		return double_tree_ext;				
	}	
	
	@Test
	public void testPropagateDataByChangedRule(){
		CGMBARTTreeNode double_tree_ext = buildDoubleTreeExt();
		assertEquals(2, double_tree_ext.left.left.pAdj());
		assertEquals(2, double_tree_ext.left.left.nAdj());
		assertEquals(9,double_tree_ext.left.left.left.sumResponses(),0);
		assertEquals(2,double_tree_ext.left.left.right.sumResponses(),0);
		//TODO
	}	
	
	//need soooo much more stuff here
	//pruneTreeAt
	//getPrunableNodes
	//getTerminalNodesWithDataAboveOrEqualToN
	//evaluate
	//splitValuesRepeated
	//numTimesAttrUsed
	
}
