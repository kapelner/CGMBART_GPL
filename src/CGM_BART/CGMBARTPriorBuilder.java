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

import java.util.ArrayList;

import CGM_Statistics.CGMTreePriorBuilder;

public class CGMBARTPriorBuilder extends CGMTreePriorBuilder {
	
	public static double ALPHA = 0.95;
	public static double BETA = 2; //see p271 in CGM10
	
	public CGMBARTPriorBuilder(ArrayList<double[]> X_y, int p) {
		super(X_y, p);
	}	
	
	public double getAlpha() {
		return ALPHA;
	}
	
	public double getBeta() {
		return BETA;
	}	
}
