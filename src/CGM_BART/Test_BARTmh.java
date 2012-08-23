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
//	
//	@Test	
//	public void testPickGrowNode(){
//		bart.SetupGibbsSampling();
//		CGMBART_mh.N_RULE = 1;
//		
//		//make sure stumps always grown
//		CGMBARTTreeNode tree = bart.gibbs_samples_of_cgm_trees.get(0).get(0);		
//		CGMBARTTreeNode grow_node = bart.pickGrowNode(tree);	
//		assertEquals(tree, grow_node);
//		
//		//now grow it once and make sure it's never the stump and that we pick the grow correctly
//		tree.splitAttributeM = 1;
//		tree.splitValue = 20.0;
//		tree.isLeaf = false;
//		tree.left = new CGMBARTTreeNode(tree);
//		tree.right = new CGMBARTTreeNode(tree);
//		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
//		
//		HashMap<CGMBARTTreeNode, Integer> counts = new HashMap<CGMBARTTreeNode, Integer>(2);
//		counts.put(tree.left, 0);
//		counts.put(tree.right, 0);
//		for (int i = 0; i < BigN; i++){
//			grow_node = bart.pickGrowNode(tree);
//			counts.put(grow_node, counts.get(grow_node) + 1);
//			assertTrue(tree != grow_node);
//		}
//		assertEquals(counts.get(tree.left) / BigN, 0.5, 0.001);
//		assertEquals(counts.get(tree.right) / BigN, 0.5, 0.001);
//		
//		//now grow it again
//		tree.right.splitAttributeM = 0;
//		tree.right.splitValue = 0.0;
//		tree.right.isLeaf = false;
//		tree.right.left = new CGMBARTTreeNode(tree.right);
//		tree.right.right = new CGMBARTTreeNode(tree.right);
//		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
//		
//		counts = new HashMap<CGMBARTTreeNode, Integer>(3);
//		counts.put(tree.left, 0);
//		counts.put(tree.right.left, 0);
//		counts.put(tree.right.right, 0);		
//		for (int i = 0; i < BigN; i++){
//			grow_node = bart.pickGrowNode(tree);
//			counts.put(grow_node, counts.get(grow_node) + 1);
//			assertTrue(tree != grow_node);
//			assertTrue(tree.right != grow_node);
//		}
//		assertEquals(counts.get(tree.left) / BigN, 0.33333, 0.01);
//		assertEquals(counts.get(tree.right.left) / BigN, 0.33333, 0.01);
//		assertEquals(counts.get(tree.right.right) / BigN, 0.33333, 0.01);
//	}	
//	
//	@Test	
//	public void testPickPruneNode(){
//		bart.SetupGibbsSampling();
//		CGMBART_mh.N_RULE = 1;
//		
//		//make sure stumps always grown
//		CGMBARTTreeNode tree = bart.gibbs_samples_of_cgm_trees.get(0).get(0);		
//		CGMBARTTreeNode prune_node = bart.pickPruneNode(tree);
//		assertNull(prune_node);
//		
//		//now grow it once and make sure it's never the stump and that we pick the grow correctly
//		tree.splitAttributeM = 1;
//		tree.splitValue = 20.0;
//		tree.isLeaf = false;
//		tree.left = new CGMBARTTreeNode(tree);
//		tree.right = new CGMBARTTreeNode(tree);
//		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
//		
//		
//		for (int i = 0; i < BigN; i++){
//			prune_node = bart.pickPruneNode(tree);
//			assertEquals(tree, prune_node);
//		}
//		
//		//now grow it again
//		tree.right.splitAttributeM = 0;
//		tree.right.splitValue = 0.0;
//		tree.right.isLeaf = false;
//		tree.right.left = new CGMBARTTreeNode(tree.right);
//		tree.right.right = new CGMBARTTreeNode(tree.right);
//		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
//				
//		for (int i = 0; i < BigN; i++){
//			prune_node = bart.pickPruneNode(tree);
//			assertEquals(tree.right, prune_node);
//		}
//		
//		
//		//now grow it again
//		tree.right.right.splitAttributeM = 2;
//		tree.right.right.splitValue = 0.0;
//		tree.right.right.isLeaf = false;
//		tree.right.right.left = new CGMBARTTreeNode(tree.right.right);
//		tree.right.right.right = new CGMBARTTreeNode(tree.right.right);
//		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
//				
//		for (int i = 0; i < BigN; i++){
//			prune_node = bart.pickPruneNode(tree);
//			assertEquals(tree.right.right, prune_node);
//		}	
//		
//		
//		
//		//now grow it again so there's two prune nodes
//		tree.left.splitAttributeM = 2;
//		tree.left.splitValue = 0.0;
//		tree.left.isLeaf = false;
//		tree.left.left = new CGMBARTTreeNode(tree.left);
//		tree.left.right = new CGMBARTTreeNode(tree.left);
//		CGMBARTTreeNode.propagateDataByChangedRule(tree, true);
//				
//		HashMap<CGMBARTTreeNode, Integer> counts = new HashMap<CGMBARTTreeNode, Integer>(2);
//		counts.put(tree.left, 0);
//		counts.put(tree.right.right, 0);		
//		for (int i = 0; i < BigN; i++){
//			prune_node = bart.pickPruneNode(tree);
//			counts.put(prune_node, counts.get(prune_node) + 1);
//		}	
//		assertEquals(counts.get(tree.left) / BigN, 0.5, 0.001);
//		assertEquals(counts.get(tree.right.right) / BigN, 0.5, 0.001);		
//	}

	@Test	
	public void testCalcLnTransRatioGrowAndPrune(){
		//we're going to test the stump's transition ratio
		CGMBARTTreeNode T_i = Test_CGMBARTTreeNode.stump;
		CGMBARTTreeNode T_star = Test_CGMBARTTreeNode.stump.clone();
		CGMBARTTreeNode first_node_grown = T_star;
		assertEquals(T_i.numLeaves(), 1); //b = 1
		assertEquals(first_node_grown.pAdj(), 3); //padj = 3
		
		//set up the stump as a tree
		first_node_grown.splitAttributeM = 0;
		assertEquals(first_node_grown.nAdj(), 4); //n_adj = 4 since there are 4 zeroes
		first_node_grown.splitValue = 0.0;
		assertEquals(first_node_grown.splitValuesRepeated(), 4); //n_repeat = 4 since the zero is repeated 4 times
		first_node_grown.isLeaf = false;
		first_node_grown.left = new CGMBARTTreeNode(T_star);
		first_node_grown.right = new CGMBARTTreeNode(T_star);
		CGMBARTTreeNode.propagateDataByChangedRule(first_node_grown, true);
		assertEquals(1, T_star.numPruneNodesAvailable()); //w_2^* = 1
		
		double trans_ratio = Math.exp(bart.calcLnTransRatioGrow(T_i, T_star, first_node_grown));
		assertEquals(3, trans_ratio, 0.0001);
		
		//now go further and test again the grow function
		T_i = T_star;
		T_star = T_star.clone();
		CGMBARTTreeNode second_node_grown = T_star.left;
		assertEquals(T_i.numLeaves(), 2); //b = 2
		System.out.println("second_node_grown.pAdj()");
		assertEquals(second_node_grown.pAdj(), 2); //padj = 2
		assertEquals(T_star.numPruneNodesAvailable(), 1); //w_2^* = 1
		second_node_grown.splitAttributeM = 2;
		assertEquals(second_node_grown.nAdj(), 2); //n_adj = 2 since there are 2 zeroes
		second_node_grown.splitValue = 0.0;
		assertEquals(second_node_grown.splitValuesRepeated(), 2); //n_repeat = 2 since the zero is repeated 2 times at this juncture
		second_node_grown.isLeaf = false;
		second_node_grown.left = new CGMBARTTreeNode(T_star);
		second_node_grown.right = new CGMBARTTreeNode(T_star);		
		CGMBARTTreeNode.propagateDataByChangedRule(T_star, true);
		
		trans_ratio = Math.exp(bart.calcLnTransRatioGrow(T_i, T_star, second_node_grown));
		assertEquals(4, trans_ratio, 0.0001);	
		
		//now test the prune function
		CGMBARTTreeNode temp = T_i;
		T_i = T_star;
		T_star = temp;
		CGMBARTTreeNode prune_node = second_node_grown;
		
		assertEquals(T_i.numPruneNodesAvailable(), 1); //w_2 = 1
		assertEquals(T_i.numLeaves(), 3); //b = 3
		assertEquals(prune_node.pAdj(), 2); //padj = 2
		assertEquals(prune_node.nAdj(), 2); //n_adj = 2 since there are 2 zeroes
		assertEquals(prune_node.splitValuesRepeated(), 2); //n_repeat = 2
		
		trans_ratio = Math.exp(bart.calcLnTransRatioPrune(T_i, T_star, prune_node));
		assertEquals(0.25, trans_ratio, 0.0001);	
		CGMBARTTreeNode.pruneTreeAt(prune_node);
		
		//prune it again
		T_i = T_star;
		T_star = Test_CGMBARTTreeNode.stump;
		prune_node = first_node_grown;
		
		assertEquals(T_i.numPruneNodesAvailable(), 1); //w_2 = 1
		assertEquals(T_i.numLeaves(), 2); //b = 2
		assertEquals(prune_node.pAdj(), 3); //padj = 3
		assertEquals(prune_node.nAdj(), 4); //n_adj = 4 since there are 4 zeroes
		assertEquals(prune_node.splitValuesRepeated(), 4); //n_repeat = 4	
		
		trans_ratio = Math.exp(bart.calcLnTransRatioPrune(T_i, T_star, prune_node));
		assertEquals(0.333333, trans_ratio, 0.0001);	
		CGMBARTTreeNode.pruneTreeAt(prune_node);	
	}

	@Test	
	public void testCalcLnTreeStructureRatioGrow(){
		//we're going to test the stump's transition ratio
		CGMBARTTreeNode T_i = Test_CGMBARTTreeNode.stump;
		CGMBARTTreeNode T_star = Test_CGMBARTTreeNode.stump.clone();
		CGMBARTTreeNode first_node_grown = T_star;
		assertEquals(first_node_grown.pAdj(), 3); //padj = 3
		
		//set up the stump as a tree
		first_node_grown.splitAttributeM = 0;
		assertEquals(first_node_grown.nAdj(), 4); //n_adj = 4 since there are 4 zeroes
		first_node_grown.splitValue = 0.0;
		assertEquals(first_node_grown.splitValuesRepeated(), 4); //n_repeat = 4 since the zero is repeated 4 times
		first_node_grown.isLeaf = false;
		first_node_grown.left = new CGMBARTTreeNode(T_star);
		first_node_grown.right = new CGMBARTTreeNode(T_star);
		CGMBARTTreeNode.propagateDataByChangedRule(first_node_grown, true);
		assertEquals(0, first_node_grown.generation); //d_eta = 0
		
		double tree_structure_ratio = Math.exp(bart.calcLnTreeStructureRatioGrow(first_node_grown));
		assertEquals(3.6822395, tree_structure_ratio, 0.0001);
		
		//now go further and test again the grow function
		T_i = T_star;
		T_star = T_star.clone();
		CGMBARTTreeNode second_node_grown = T_star.left;
		assertEquals(second_node_grown.pAdj(), 2); //padj = 2
		assertEquals(1, second_node_grown.generation); //d_eta = 1
		second_node_grown.splitAttributeM = 2;
		assertEquals(second_node_grown.nAdj(), 2); //n_adj = 2 since there are 2 zeroes
		second_node_grown.splitValue = 0.0;
		assertEquals(second_node_grown.splitValuesRepeated(), 2); //n_repeat = 2 since the zero is repeated 2 times at this juncture
		second_node_grown.isLeaf = false;
		second_node_grown.left = new CGMBARTTreeNode(T_star);
		second_node_grown.right = new CGMBARTTreeNode(T_star);		
		CGMBARTTreeNode.propagateDataByChangedRule(T_star, true);
		
		tree_structure_ratio = Math.exp(bart.calcLnTreeStructureRatioGrow(second_node_grown));
		assertEquals(0.124594970654, tree_structure_ratio, 0.0001);	
		
		//now test the prune function
		CGMBARTTreeNode temp = T_i;
		T_i = T_star;
		T_star = temp;
		CGMBARTTreeNode prune_node = second_node_grown;
		
		assertEquals(prune_node.pAdj(), 2); //padj = 2
		assertEquals(prune_node.nAdj(), 2); //n_adj = 2 since there are 2 zeroes
		assertEquals(prune_node.splitValuesRepeated(), 2); //n_repeat = 2
		assertEquals(1, prune_node.generation);
		
		
		tree_structure_ratio = Math.exp(-bart.calcLnTreeStructureRatioGrow(prune_node));
		assertEquals(8.02600614418, tree_structure_ratio, 0.0001);	
		CGMBARTTreeNode.pruneTreeAt(prune_node);
		
		//prune it again
		T_i = T_star;
		T_star = Test_CGMBARTTreeNode.stump;
		prune_node = first_node_grown;
		
		assertEquals(prune_node.generation, 0); //padj = 3
		assertEquals(prune_node.pAdj(), 3); //padj = 3
		assertEquals(prune_node.nAdj(), 4); //n_adj = 4 since there are 4 zeroes
		assertEquals(prune_node.splitValuesRepeated(), 4); //n_repeat = 4	
		
		tree_structure_ratio = Math.exp(-bart.calcLnTreeStructureRatioGrow(prune_node));
		assertEquals(0.271573855, tree_structure_ratio, 0.0001);	
		CGMBARTTreeNode.pruneTreeAt(prune_node);	
	}

	@Test	
	public void testCalcLnLikRatioGrow(){
		//first set it up so sigsq = 0.01
		double sigsq = 0.01;
		bart.gibb_sample_num = 1;
		bart.InitGibbsSamplingData();
		bart.gibbs_samples_of_sigsq.add(0.01);
		
		//we're going to test the stump's transition ratio
		CGMBARTTreeNode T_i = Test_CGMBARTTreeNode.stump;
		CGMBARTTreeNode T_star = Test_CGMBARTTreeNode.stump.clone();
		CGMBARTTreeNode first_node_grown = T_star;
		assertEquals(784, first_node_grown.sumResponsesQuantitySqd(), 0.001);
		
		//make the node grow
		first_node_grown.splitAttributeM = 0;
		first_node_grown.splitValue = 0.0;
		first_node_grown.isLeaf = false;
		first_node_grown.left = new CGMBARTTreeNode(T_star);
		first_node_grown.right = new CGMBARTTreeNode(T_star);
		CGMBARTTreeNode.propagateDataByChangedRule(first_node_grown, true);
		
		//now ensure the proper n's
		assertEquals(7, first_node_grown.n_eta);
		assertEquals(4, first_node_grown.left.n_eta);
		assertEquals(3, first_node_grown.right.n_eta);
		assertEquals(256, first_node_grown.left.sumResponsesQuantitySqd(), 0.001);
		assertEquals(144, first_node_grown.right.sumResponsesQuantitySqd(), 0.001);
		
		double sigsq_plus_n_ell_hyper_sisgsq_mu = sigsq + 7 * bart.hyper_sigsq_mu;
		double sigsq_plus_n_ell_L_hyper_sisgsq_mu = sigsq + 4 * bart.hyper_sigsq_mu;
		double sigsq_plus_n_ell_R_hyper_sisgsq_mu = sigsq + 3 * bart.hyper_sigsq_mu;
		double c = 0.5 * (
				Math.log(sigsq) 
				+ Math.log(sigsq_plus_n_ell_hyper_sisgsq_mu) 
				- Math.log(sigsq_plus_n_ell_L_hyper_sisgsq_mu) 
				- Math.log(sigsq_plus_n_ell_R_hyper_sisgsq_mu));
		double d = bart.hyper_sigsq_mu / (2 * sigsq);
		double e = 256 / sigsq_plus_n_ell_L_hyper_sisgsq_mu
				+ 144 / sigsq_plus_n_ell_R_hyper_sisgsq_mu
				- 784 / sigsq_plus_n_ell_hyper_sisgsq_mu;		
		
		double lik_ratio = Math.exp(bart.calcLnLikRatioGrow(first_node_grown));
		double first_split_lik_ratio = Math.exp(c + d * e);
		assertEquals(first_split_lik_ratio, lik_ratio, 0.0001);
		
		//now go further and test again the grow function
		T_i = T_star;
		T_star = T_star.clone();
		//grow again 
		CGMBARTTreeNode second_node_grown = T_star.left;
		second_node_grown.splitAttributeM = 2;
		second_node_grown.splitValue = 0.0;
		second_node_grown.isLeaf = false;
		second_node_grown.left = new CGMBARTTreeNode(T_star);
		second_node_grown.right = new CGMBARTTreeNode(T_star);		
		CGMBARTTreeNode.propagateDataByChangedRule(T_star, true);
		
		//now ensure the proper n's
		assertEquals(4, second_node_grown.n_eta);
		assertEquals(2, second_node_grown.left.n_eta);
		assertEquals(2, second_node_grown.right.n_eta);
		assertEquals(196, second_node_grown.left.sumResponsesQuantitySqd(), 0.001);
		assertEquals(4, second_node_grown.right.sumResponsesQuantitySqd(), 0.001);	
		
		sigsq_plus_n_ell_hyper_sisgsq_mu = sigsq + 4 * bart.hyper_sigsq_mu;
		sigsq_plus_n_ell_L_hyper_sisgsq_mu = sigsq + 2 * bart.hyper_sigsq_mu;
		sigsq_plus_n_ell_R_hyper_sisgsq_mu = sigsq + 2 * bart.hyper_sigsq_mu;
		c = 0.5 * (
				Math.log(sigsq) 
				+ Math.log(sigsq_plus_n_ell_hyper_sisgsq_mu) 
				- Math.log(sigsq_plus_n_ell_L_hyper_sisgsq_mu) 
				- Math.log(sigsq_plus_n_ell_R_hyper_sisgsq_mu));
		d = bart.hyper_sigsq_mu / (2 * sigsq);
		e = 192 / sigsq_plus_n_ell_L_hyper_sisgsq_mu
				+ 4 / sigsq_plus_n_ell_R_hyper_sisgsq_mu
				- 256 / sigsq_plus_n_ell_hyper_sisgsq_mu;
		
		
		lik_ratio = Math.exp(bart.calcLnLikRatioGrow(second_node_grown));
		double second_split_lik_ratio = Math.exp(c + d * e);
		assertEquals(second_split_lik_ratio, lik_ratio, 0.0001);	
		
		//now test the prune function
		CGMBARTTreeNode temp = T_i;
		T_i = T_star;
		T_star = temp;
		CGMBARTTreeNode prune_node = second_node_grown;
		
		
		lik_ratio = Math.exp(-bart.calcLnLikRatioGrow(prune_node));
		assertEquals(Math.pow(second_split_lik_ratio, -1), lik_ratio, 0.0001);	//inverse of grow
		CGMBARTTreeNode.pruneTreeAt(prune_node);
		
		//prune it again
		T_i = T_star;
		T_star = Test_CGMBARTTreeNode.stump;
		prune_node = first_node_grown;
		
		lik_ratio = Math.exp(-bart.calcLnLikRatioGrow(prune_node));
		assertEquals(Math.pow(first_split_lik_ratio, -1), lik_ratio, 0.0001);	//inverse of grow	
		CGMBARTTreeNode.pruneTreeAt(prune_node);
	}

}