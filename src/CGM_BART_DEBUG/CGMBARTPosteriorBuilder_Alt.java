package CGM_BART_DEBUG;

import CGM_BART.*;
import CGM_BayesianCART1998.CGMPosteriorBuilder.Steps;
import CGM_Statistics.CGMTreePriorBuilder;

/**
 * Only change step is done
 * 
 * @author kapelner
 *
 */
public class CGMBARTPosteriorBuilder_Alt extends CGMBARTPosteriorBuilderOld {

	public CGMBARTPosteriorBuilder_Alt(CGMTreePriorBuilder tree_prior_builder) {
		super(tree_prior_builder);
	}
	
	protected Steps randomlyPickAmongTheFourProposalSteps() {
//		double roll = Math.random();
//		if (roll < 0.01)
//			return Steps.GROW;
//		else if (roll < 0.02)
//			return Steps.PRUNE;
//		if (roll <= 1)
//			return Steps.CHANGE;
//		return Steps.SWAP;
		
		return Steps.CHANGE;
	}	
}
