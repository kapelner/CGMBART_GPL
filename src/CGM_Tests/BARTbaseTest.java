package CGM_Tests;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import CGM_BART.CGMBART_eval;

public class BARTbaseTest {

	private static CGMBART_eval simple_bart;


	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		simple_bart = TreeTest.bart;
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
	public void testMetadataCreation(){
		double[] min_vals = {0, 9.3, 0};
		assertArrayEquals(simple_bart.getMinimum_values_by_attribute(), min_vals, 0);
		double[] max_vals = {1, 65.9, 1};
		assertArrayEquals(simple_bart.getMaximum_values_by_attribute(), max_vals, 0);		
		//check freq
		assertEquals(simple_bart.frequencyValueForAttribute(0, 0.0), 4);
		assertEquals(simple_bart.frequencyValueForAttribute(0, 1), 3);
		assertEquals(simple_bart.frequencyValueForAttribute(1, 15.3), 1);
		assertEquals(simple_bart.frequencyValueForAttribute(1, 45.8), 1);
		assertEquals(simple_bart.frequencyValueForAttribute(1, 31.2), 1);
		assertEquals(simple_bart.frequencyValueForAttribute(1, 65.9), 1);
		assertEquals(simple_bart.frequencyValueForAttribute(1, 32.3), 1);
		assertEquals(simple_bart.frequencyValueForAttribute(1, 9.3), 2);
		assertEquals(simple_bart.frequencyValueForAttribute(1, 15.3), 1);
		assertEquals(simple_bart.frequencyValueForAttribute(2, 0), 3);
		assertEquals(simple_bart.frequencyValueForAttribute(2, 1), 4);		
	}

	
	@Test 
	public void testHyperparams(){
			
	}
}