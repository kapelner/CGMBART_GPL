package CGM_BART;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;



public class Test_BARTinit {

	private static CGMBART_init bart;


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
	public void testSetupGibbsSampling(){
		bart.SetupGibbsSampling();
		assertEquals(bart.gibbs_samples_of_cgm_trees.size(), 1);
	}
}