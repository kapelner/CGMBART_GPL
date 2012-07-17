package CGM_Tests;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import CGM_BART.CGMBARTPriorBuilder;
import CGM_Statistics.CGMTreeNode;

public class PriorTest {

	public static CGMBARTPriorBuilder simple_tree_prior_builder;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		simple_tree_prior_builder = new CGMBARTPriorBuilder(TreeTest.data);
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
	public void testSimpleTreePriorBuilderIntegrity(){
		simple_tree_prior_builder
	}

}
