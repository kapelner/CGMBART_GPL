package CGM_BART_DEBUG;

import java.util.ArrayList;

import CGM_BART.CGMBART;
import CGM_BART.CGMBARTPosteriorBuilder;
import CGM_BART.CGMBARTTreeNode;
import CGM_Statistics.StatToolbox;

public class CGMBARTPosteriorBuilder_OnlyParent extends CGMBARTPosteriorBuilder {

	public CGMBARTPosteriorBuilder_OnlyParent(CGMBART cgmbart) {
		super(cgmbart);
	}

	
	protected CGMBARTTreeNode pickChangeNode(ArrayList<CGMBARTTreeNode> internal_nodes) {
		System.out.println("better pickChangeNode");
		ArrayList<CGMBARTTreeNode> better_internal_nodes = new ArrayList<CGMBARTTreeNode>();
		for (CGMBARTTreeNode node : internal_nodes){
			int d = node.getGeneration();
			for (int i = 0; i < (int)Math.round(100 * 1 / Math.pow(d + 1, 2)); i++){ //(double)(d + 1)
				better_internal_nodes.add(node);
			}
		}
		//return a random one
		return better_internal_nodes.get(((int)Math.floor(StatToolbox.rand() * better_internal_nodes.size())));
	}
	
	//always reject so we can see what's going on
//	protected boolean acceptOrRejectProposal(double ln_u_0_1, double log_r){
//		return false;
//	}	

	
}
