package CGM_BART_DEBUG;

import java.util.ArrayList;

public class CGMBART_FixedTreeStructureChangeRulesAndSigsqOnlyParent extends CGMBART_FixedTreeStructureChangeRulesAndSigsq {
	private static final long serialVersionUID = 3460935328647793073L;	
		
	public CGMBART_FixedTreeStructureChangeRulesAndSigsqOnlyParent() {
		super();
		setNumTrees(1); //in this DEBUG model, there's only one tree
	}
	
	public void setData(ArrayList<double[]> X_y){
		super.setData(X_y);
		fixed_sigsq = 1 / y_range_sq;
	}

}
