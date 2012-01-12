/*
    BART - Bayesian Additive Regressive Trees
    Software for Supervised Statistical Learning
    
    Copyright (C) 2012 Professor Ed George & Adam Kapelner, 
    Dept of Statistics, The Wharton School of the University of Pennsylvania

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details:
    
    http://www.gnu.org/licenses/gpl-2.0.txt

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

package CGM_BayesianCART1998;

import CGM_Statistics.*;

import GemIdentClassificationEngine.DatumSetupForEntireRun;
import GemIdentView.JProgressBarAndLabel;

public class CGMRegressionTree extends CGMCART implements LeafAssigner {
	private static final long serialVersionUID = -8440435386881914334L;
	
	public CGMRegressionTree(DatumSetupForEntireRun datumSetupForEntireRun, JProgressBarAndLabel buildProgress) {
		super(datumSetupForEntireRun, buildProgress);
	}

	@Override
	protected void createPosteriorBuilder() {
		tree_posterior_builder = new CGMRegressionMeanShiftPosteriorPlusEnhBuilder(tree_prior_builder, y);	
	}
	
	@Override
	public void assignLeaf(CGMTreeNode node) {
		//for now just take mean of the y's
		double[] y_is = node.get_ys_in_data();
		node.y_prediction = StatToolbox.sample_average(y_is);
	}
	
}