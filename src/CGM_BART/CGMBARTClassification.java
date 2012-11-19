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

package CGM_BART;

public final class CGMBARTClassification extends CGMBART_F1_prior_cov_spec {
	private static final long serialVersionUID = -9061432248755912576L;
	
	/** the number of classes */
//	private Integer K;
	/**
	 * Constructs the BART classifier for classification. We rely on the SetupClassification class to set the raw data
	 * 
	 * @param datumSetup
	 * @param buildProgress
	 */
	public CGMBARTClassification(int K) {
		super();
//		this.K = K;
	}	
	
	@Override
	public double Evaluate(double[] record) {
		return InverseProbit(super.Evaluate(record));
	}	
	

	private double InverseProbit(double y_star) {
		// TODO Auto-generated method stub
		return y_star;
	}
	

	@Override
	protected void DoGibbsSampling() {
		// TODO Auto-generated method stub
		
	}
}
