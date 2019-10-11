package com.camoga.examples.cnn;

import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class NormalizeImages {
	public static void main(String[] args) {
		int imgoutput = 0;
		for(int i = 0; i < 12500; i++) {
			BufferedImage image = null;
			try {
				image = ImageIO.read(NormalizeImages.class.getResourceAsStream("/CNN/catdog/PetImages/Dog/"+i+".jpg"));
			} catch (IOException e) {
				continue;
			}
			if(image == null) continue;
			int w = image.getWidth();
			int h = image.getHeight();
			
			AffineTransform at = new AffineTransform();
			double min = Math.min(w, h);
			at.scale(100/min, 100/min);
			
			AffineTransformOp op = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
			BufferedImage output = new BufferedImage(100,100 ,image.getType());
			output = op.filter(image, output);
			System.out.println(output.getWidth() +","+output.getHeight());
			try {
				ImageIO.write(output, "png", new File(imgoutput+".png"));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			System.out.println(i);
			imgoutput++;
		}
	}
}
