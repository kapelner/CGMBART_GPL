package CGM_Tests;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;


import CGM_BART.CGMBART_eval;

public class BARThyperparamsTest {

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
	
	

	
	@Test //y = {0, 0, 2, 4, 5, 8, 9};
	public void testTransformAndUnVariable(){
		assertEquals(simple_bart.getY_min(), 0, 0);
		assertEquals(simple_bart.getY_max(), 9, 0);
		assertEquals(simple_bart.getY_range_sq(), 9, 81);
		double[] ytrans = {-0.5, -0.5, -0.27777, -0.055555, 0.055555, 0.38888, 0.5};
		assertArrayEquals(simple_bart.getYTrans(), ytrans, 0.0001);
		assertArrayEquals(simple_bart.un_transform_y(ytrans), TreeTest.y, 0.0001);
	}
}