package CGM_BART;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class Test_BARTgibbs_base {

	private static CGMBART_gibbs_base bart;
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
	public void testBuildCorrectDims(){
		bart.Build();
		assertEquals(NGAndNB + 1, bart.gibb_sample_num); //remember how for loops work? If we want this to end
		assertEquals(NGAndNB + 1, bart.gibbs_samples_of_cgm_trees.size());
		for (int i = 1; i <= NGAndNB; i++){
			assertEquals(bart.num_trees, bart.gibbs_samples_of_cgm_trees.get(i).size());
		}		
		assertEquals(NGAndNB + 1, bart.gibbs_samples_of_sigsq.size());
		assertEquals(NGAndNB - NB, bart.gibbs_samples_of_cgm_trees_after_burn_in.size());
		assertEquals(NGAndNB - NB, bart.gibbs_samples_of_sigsq_after_burn_in.size());
	}
	
	
	@Test
	public void testSampleSigsq(){
		bart.Build();
		for (int i = 1; i <= NGAndNB; i++){
			bart.SampleSigsq(i);
			assertTrue(bart.gibbs_samples_of_sigsq.get(i) > 0);
		}
	}	
	
}