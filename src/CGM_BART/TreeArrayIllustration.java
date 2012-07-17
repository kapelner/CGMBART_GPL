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

import java.awt.image.BufferedImage;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;

import javax.media.jai.JAI;

import CGM_Statistics.*;

public class TreeArrayIllustration {

	private int sample_num;
	private ArrayList<CGMTreeNode> trees;
	private ArrayList<Double> likelihoods;
	
	public static NumberFormat one_digit_format = NumberFormat.getInstance();
	static {
		one_digit_format.setMaximumFractionDigits(1);
	}	

	public TreeArrayIllustration(int sample_num) {
		this.sample_num = sample_num;
		trees = new ArrayList<CGMTreeNode>();
		likelihoods = new ArrayList<Double>();
	}

	public void AddTree(CGMTreeNode tree) {
		trees.add(tree);
	}

	public void addLikelihood(double lik) {
		likelihoods.add(lik);
	}	
	
	public void CreateIllustrationAndSaveImage() {
		//first pull out all the tree images
		int m = trees.size();
		int w = 0;
		int h = Integer.MIN_VALUE;
		ArrayList<BufferedImage> canvases = new ArrayList<BufferedImage>(m);
		for (int t = 0; t < m; t++){
			CGMTreeNode tree = trees.get(t);
			HashMap<String, String> info = new HashMap<String, String>();
			info.put("tree_num", "" + (t + 1));
			info.put("num_iteration", "" + sample_num);
			info.put("likelihood", "" + one_digit_format.format(likelihoods.get(t)));
			BufferedImage canvas = new TreeIllustration(tree, info).getCanvas();
			w += canvas.getWidth(); //aggregate the widths
			if (canvas.getHeight() > h){ //get the maximum height
				h = canvas.getHeight();
			}
			canvases.add(canvas);
		}
		
		BufferedImage master_canvas = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_BINARY);
		int sliding_width = 0;
		for (int t = 0; t < m; t++){
			BufferedImage canvas = canvases.get(t);
			master_canvas.getGraphics().drawImage(canvas, sliding_width, 0, null);
			sliding_width += canvas.getWidth();
		}
		saveImageFile(master_canvas);
		
	}
	
	private void saveImageFile(BufferedImage image) {
		String title = "BART_trees_iter_" + LeadingZeroes(sample_num, 5);
		JAI.create("filestore", image, CGMShared.DEBUG_DIR + "//" + title + ".png", "PNG");
	}
	
	private static final String ZEROES = "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
	public static String LeadingZeroes(double num, int num_digits) {
		if (num < 10 && num_digits >= 2){
			return ZEROES.substring(0, num_digits - 1) + num;
		}
		else if (num < 100 && num_digits >= 3){
			return ZEROES.substring(0, num_digits - 2) + num;
		}
		else if (num < 1000 && num_digits >= 4){
			return ZEROES.substring(0, num_digits - 3) + num;
		}
		else if (num < 10000 && num_digits >= 5){
			return ZEROES.substring(0, num_digits - 4) + num;
		}
		else if (num < 100000 && num_digits >= 6){
			return ZEROES.substring(0, num_digits - 5) + num;
		}
		else if (num < 1000000 && num_digits >= 7){
			return ZEROES.substring(0, num_digits - 6) + num;
		}
		else if (num < 10000000 && num_digits >= 8){
			return ZEROES.substring(0, num_digits - 7) + num;
		}
		return String.valueOf(num);
	}
	public static String LeadingZeroes(int num, int num_digits) {
		if (num < 10 && num_digits >= 2){
			return ZEROES.substring(0, num_digits - 1) + num;
		}
		else if (num < 100 && num_digits >= 3){
			return ZEROES.substring(0, num_digits - 2) + num;
		}
		else if (num < 1000 && num_digits >= 4){
			return ZEROES.substring(0, num_digits - 3) + num;
		}
		else if (num < 10000 && num_digits >= 5){
			return ZEROES.substring(0, num_digits - 4) + num;
		}
		else if (num < 100000 && num_digits >= 6){
			return ZEROES.substring(0, num_digits - 5) + num;
		}
		else if (num < 1000000 && num_digits >= 7){
			return ZEROES.substring(0, num_digits - 6) + num;
		}
		else if (num < 10000000 && num_digits >= 8){
			return ZEROES.substring(0, num_digits - 7) + num;
		}
		return String.valueOf(num);
	}		


	
}
