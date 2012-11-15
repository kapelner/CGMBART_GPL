package CGM_BART;

import static org.junit.Assert.*;

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
	
	@Test
	public void testInitializeMus(){
		bart.InitGibbsSamplingData();
		bart.InitizializeSigsq();
		bart.InitializeTrees();
		bart.InitializeMus();
		CGMBARTTreeNode tree = bart.gibbs_samples_of_cgm_trees[0][0];
		assertTrue(tree.y_pred == CGMBART_04_init.INITIAL_PRED);
	}	
}