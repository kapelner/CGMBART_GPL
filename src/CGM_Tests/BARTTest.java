package CGM_Tests;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;

import CGM_BART.CGMBARTTreeNode;

public class BARTTest {

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
	public void testSimpleConstructor(){
		double[] min_vals = {0, 9.3, 0};
		assertArrayEquals(simple_tree.getMinimumValuesByAttribute(), min_vals, 0);
		//check freq
		assertEquals(pb.frequencyValueForAttribute(0, 0), 4);
		assertEquals(pb.frequencyValueForAttribute(0, 1), 3);
		assertEquals(pb.frequencyValueForAttribute(1, 15.3), 1);
		assertEquals(pb.frequencyValueForAttribute(1, 45.8), 1);
		assertEquals(pb.frequencyValueForAttribute(1, 31.2), 1);
		assertEquals(pb.frequencyValueForAttribute(1, 65.9), 1);
		assertEquals(pb.frequencyValueForAttribute(1, 32.3), 1);
		assertEquals(pb.frequencyValueForAttribute(1, 9.3), 2);
		assertEquals(pb.frequencyValueForAttribute(1, 15.3), 1);
		assertEquals(pb.frequencyValueForAttribute(2, 0), 3);
		assertEquals(pb.frequencyValueForAttribute(2, 1), 4);		
	}
	
}