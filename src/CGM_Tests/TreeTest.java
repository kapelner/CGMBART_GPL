package CGM_Tests;

import static org.junit.Assert.*;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;



import CGM_Statistics.CGMTreeNode;

public class TreeTest {

	public static double[] y = {0, 0, 2, 4, 5, 8, 9};
	public static ArrayList<double[]> data;
	public static CGMTreeNode stump;	
	public static CGMTreeNode simple_tree;
	
	static {
		data = new ArrayList<double[]>();
		double[] x_0 = {0, 1, 0, 1, 0, 1, 0};
		double[] x_1 = {15.3, 45.8, 31.2, 65.9, 49.1, 32.3, 9.3};
		double[] x_2 = {1, 1, 1, 1, 0, 0, 0};
		for (int i = 0; i < x_1.length; i++){
			double datum[] = {x_0[i], x_1[i], x_2[i], y[i]};
			data.add(datum);
		}	
		stump = new CGMTreeNode(null, data, null);
		
		simple_tree = new CGMTreeNode(null, data, null);
		simple_tree.splitAttributeM = 0;
		simple_tree.splitValue = 0.0;
		simple_tree.isLeaf = false;
		simple_tree.left = new CGMTreeNode(simple_tree, null);
		simple_tree.right = new CGMTreeNode(simple_tree, null);
		CGMTreeNode.propagateRuleChangeOrSwapThroughoutTree(simple_tree, true);		
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
		CGMTreeNode cloned_stump = stump.clone();
		assertEquals(cloned_stump.n, stump.n);
		assertArrayEquals(stump.responses(), cloned_stump.responses(), 0);
		for (int i = 0; i < stump.n; i++){
			assertArrayEquals(stump.data.get(i), cloned_stump.data.get(i), 0);
		}
		assertTrue(cloned_stump.isLeaf);
	}	
	
	@Test
	public void testTerminalNodesStump() {
		ArrayList<CGMTreeNode> just_stump = new ArrayList<CGMTreeNode>();
		just_stump.add(stump);
		assertEquals(stump.getTerminalNodes(), just_stump);
		assertEquals(CGMTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 5), just_stump);
		assertEquals(CGMTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 7), just_stump);
		assertTrue(CGMTreeNode.getTerminalNodesWithDataAboveOrEqualToN(stump, 8).size() == 0);
	}
	
	@Test 
	public void testSimpleTreeIntegrity(){
		assertEquals(simple_tree.isStump(), false);
		assertEquals(simple_tree.numLeaves(), 2);
		assertEquals(simple_tree.numPruneNodesAvailable(), 1);
		assertEquals(simple_tree.deepestNode(), 1);
		assertEquals(simple_tree.widestGeneration(), 2);
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
		ArrayList<CGMTreeNode> internal_nodes = new ArrayList<CGMTreeNode>(1);
		internal_nodes.add(simple_tree);
		assertArrayEquals(CGMTreeNode.findInternalNodes(simple_tree).toArray(), internal_nodes.toArray());
		Object[] none = new Object[0];
		assertArrayEquals(CGMTreeNode.findDoublyInternalNodes(simple_tree).toArray(), none);
	}

	
}
