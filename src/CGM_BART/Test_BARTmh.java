package CGM_BART;

import static org.junit.Assert.*;

import java.util.HashMap;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class Test_BARTmh {

	private static CGMBART_mh bart;
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
	
	private static final double BigN = 100000;
	
	@Test	
	public void testPickGrowNode(){
		bart.SetupGibbsSampling();
		CGMBART_mh.N_RULE = 1;
		
		//make sure stumps always grown
		CGMBARTTreeNode tree = bart.gibbs_samples_of_cgm_trees.get(0).get(0);		
		CGMBARTTreeNode grow_node = bart.pickGrowNode(tree);	
		assertEquals(tree, grow_node);
		
		//now grow it once and make sure it's never the stump and that we pick the grow correctly
		tree.splitAttributeM = 1;
		tree.splitValue = 20.0;
		tree.isLeaf = false;
		tree.left = new CGMBARTTreeNode(tree);
		tree.right = new CGMBARTTreeNode(tree);
		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
		
		HashMap<CGMBARTTreeNode, Integer> counts = new HashMap<CGMBARTTreeNode, Integer>(2);
		counts.put(tree.left, 0);
		counts.put(tree.right, 0);
		for (int i = 0; i < BigN; i++){
			grow_node = bart.pickGrowNode(tree);
			counts.put(grow_node, counts.get(grow_node) + 1);
			assertTrue(tree != grow_node);
		}
		assertEquals(counts.get(tree.left) / BigN, 0.5, 0.001);
		assertEquals(counts.get(tree.right) / BigN, 0.5, 0.001);
		
		//now grow it again
		tree.right.splitAttributeM = 0;
		tree.right.splitValue = 0.0;
		tree.right.isLeaf = false;
		tree.right.left = new CGMBARTTreeNode(tree.right);
		tree.right.right = new CGMBARTTreeNode(tree.right);
		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
		
		counts = new HashMap<CGMBARTTreeNode, Integer>(3);
		counts.put(tree.left, 0);
		counts.put(tree.right.left, 0);
		counts.put(tree.right.right, 0);		
		for (int i = 0; i < BigN; i++){
			grow_node = bart.pickGrowNode(tree);
			counts.put(grow_node, counts.get(grow_node) + 1);
			assertTrue(tree != grow_node);
			assertTrue(tree.right != grow_node);
		}
		assertEquals(counts.get(tree.left) / BigN, 0.33333, 0.01);
		assertEquals(counts.get(tree.right.left) / BigN, 0.33333, 0.01);
		assertEquals(counts.get(tree.right.right) / BigN, 0.33333, 0.01);
	}	
	
	@Test	
	public void testPickPruneNode(){
		bart.SetupGibbsSampling();
		CGMBART_mh.N_RULE = 1;
		
		//make sure stumps always grown
		CGMBARTTreeNode tree = bart.gibbs_samples_of_cgm_trees.get(0).get(0);		
		CGMBARTTreeNode prune_node = bart.pickPruneNode(tree);
		assertNull(prune_node);
		
		//now grow it once and make sure it's never the stump and that we pick the grow correctly
		tree.splitAttributeM = 1;
		tree.splitValue = 20.0;
		tree.isLeaf = false;
		tree.left = new CGMBARTTreeNode(tree);
		tree.right = new CGMBARTTreeNode(tree);
		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
		
		
		for (int i = 0; i < BigN; i++){
			prune_node = bart.pickPruneNode(tree);
			assertEquals(tree, prune_node);
		}
		
		//now grow it again
		tree.right.splitAttributeM = 0;
		tree.right.splitValue = 0.0;
		tree.right.isLeaf = false;
		tree.right.left = new CGMBARTTreeNode(tree.right);
		tree.right.right = new CGMBARTTreeNode(tree.right);
		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
				
		for (int i = 0; i < BigN; i++){
			prune_node = bart.pickPruneNode(tree);
			assertEquals(tree.right, prune_node);
		}
		
		
		//now grow it again
		tree.right.right.splitAttributeM = 2;
		tree.right.right.splitValue = 0.0;
		tree.right.right.isLeaf = false;
		tree.right.right.left = new CGMBARTTreeNode(tree.right.right);
		tree.right.right.right = new CGMBARTTreeNode(tree.right.right);
		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
				
		for (int i = 0; i < BigN; i++){
			prune_node = bart.pickPruneNode(tree);
			assertEquals(tree.right.right, prune_node);
		}	
		
		
		
		//now grow it again so there's two prune nodes
		tree.left.splitAttributeM = 2;
		tree.left.splitValue = 0.0;
		tree.left.isLeaf = false;
		tree.left.left = new CGMBARTTreeNode(tree.left);
		tree.left.right = new CGMBARTTreeNode(tree.left);
		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
				
		HashMap<CGMBARTTreeNode, Integer> counts = new HashMap<CGMBARTTreeNode, Integer>(2);
		counts.put(tree.left, 0);
		counts.put(tree.right.right, 0);		
		for (int i = 0; i < BigN; i++){
			prune_node = bart.pickPruneNode(tree);
			counts.put(prune_node, counts.get(prune_node) + 1);
		}	
		assertEquals(counts.get(tree.left) / BigN, 0.5, 0.001);
		assertEquals(counts.get(tree.right.right) / BigN, 0.5, 0.001);		
	}
	
}