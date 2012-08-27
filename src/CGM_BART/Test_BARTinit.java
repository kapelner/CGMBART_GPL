package CGM_BART;

import static org.junit.Assert.*;
import static org.hamcrest.core.IsNot.*;

import java.util.ArrayList;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;



public class Test_BARTinit {

	private static CGMBART_04_init bart;


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
	}

	@After
	public void tearDown() throws Exception {
	}
	
	@Test
	public void testInitGibbsSamplingData(){
		bart.InitGibbsSamplingData();
		assertEquals(bart.gibbs_samples_of_cgm_trees.size(), 0);
		assertEquals(bart.gibbs_samples_of_cgm_trees_after_burn_in.size(), 0);
		assertEquals(bart.gibbs_samples_of_sigsq.size(), 0);
		assertEquals(bart.gibbs_samples_of_sigsq_after_burn_in.size(), 0);
	}
	
	@Test
	public void testInitizializeSigsq(){
		bart.InitGibbsSamplingData();
		bart.InitizializeSigsq();
		assertEquals(bart.gibbs_samples_of_sigsq.size(), 1);
		assertTrue(bart.gibbs_samples_of_sigsq.get(0) >= 0);
	}	
	
	@Test
	public void testInitiatizeTrees(){
		bart.InitGibbsSamplingData();
		bart.InitiatizeTrees();
		assertEquals(bart.gibbs_samples_of_cgm_trees.size(), 1);
		ArrayList<CGMBARTTreeNode> trees = bart.gibbs_samples_of_cgm_trees.get(0);
		assertEquals(trees.size(), bart.num_trees);
		CGMBARTTreeNode tree = trees.get(0);
		assertTrue(tree.isStump());
		assertEquals(tree.y_prediction, 0, 0);
		assertEquals(tree.data.size(), Test_CGMBARTTreeNode.data.size());
		assertEquals(tree.data.get(0)[0], Test_CGMBARTTreeNode.data.get(0)[0], 0);
		assertEquals(tree.data.get(0)[1], Test_CGMBARTTreeNode.data.get(0)[1], 0);
		assertEquals(tree.data.get(0)[2], Test_CGMBARTTreeNode.data.get(0)[2], 0);
	}
	
	@Test
	public void testInitializeMus(){
		bart.InitGibbsSamplingData();
		bart.InitizializeSigsq();
		bart.InitiatizeTrees();
		bart.InitializeMus();
		CGMBARTTreeNode tree = bart.gibbs_samples_of_cgm_trees.get(0).get(0);
		assertThat(tree.y_prediction, not(0.0));
	}	
}