package com.camoga.cnn;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class Window extends JFrame {
	
	class Panel extends JPanel {
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);
			int x = 0;
			for(int i = 0; i < ConvNet.cnnlayers.size(); i++) {
				if(ConvNet.cnnlayers.get(i) instanceof ConvLayer) {
					BufferedImage img = createImage(((ConvLayer)ConvNet.cnnlayers.get(i)).kernel[0]);
					g.drawImage(img, x, 0, img.getWidth()*5,img.getHeight()*5,null);
					x+=80;
//					((ConvLayer)ConvNet.layers.get(i))
				}
				BufferedImage img = createImage(ConvNet.cnnlayers.get(i).output());
				g.drawImage(img, x, 0, img.getWidth(),img.getHeight(),null);
				x+=100;
			}
			
			repaint();
		}
		
		public BufferedImage createImage(double[][][] layer) {
			BufferedImage image = new BufferedImage(layer[0].length, layer[0].length*layer.length, BufferedImage.TYPE_INT_RGB);
			int[] pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
			
			for(int n = 0; n < layer.length; n++) {
				for(int y = 0; y < layer[0].length; y++) {
					for(int x = 0; x < layer[0].length; x++) {
						pixels[n*layer[0].length*layer[0].length+y*layer[0].length+x] = ((int)(layer[n][y][x])*255)*0x10101;
					}
				}
			}
			
			return image;
		}
	}
	
	public Window() {
		setSize(800,800);
		setResizable(true);
		setVisible(true);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		add(new Panel());
	}
}