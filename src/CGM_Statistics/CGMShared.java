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

package CGM_Statistics;

import CGM_BART.CGMBARTTreeNode;

public class CGMShared {
	
	
	public static final double MostOfTheDistribution = 0.95;	
	public static final String DEBUG_DIR = "debug_output";
	
	/**
	 * Assign classes to the leaves... do it recursively...
	 */
	public static void assignLeaves(CGMBARTTreeNode node, LeafAssigner leaf_assigner) {
//		System.out.println("fill in leaves b = " + CGMTreeNode.numTerminalNodes(node));
		if (node.isLeaf){
			leaf_assigner.assignLeaf(node);
		}
		else {
			assignLeaves(node.left, leaf_assigner);
			assignLeaves(node.right, leaf_assigner);
		}
	}
}
