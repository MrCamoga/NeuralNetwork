package com.camoga.nn;

import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Arrays;

public class Paint implements MouseMotionListener, MouseListener {

	public BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
	public int[] pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
	public int scale = 3;
	public int xo = 500, yo = 250;
	
	public Paint() {
		Thread thread = new Thread(() -> run(), "Paint"); 
		thread.start();
	}
	
	public void run() {
		
		long lastTime = System.nanoTime();
		
		double ns = 1e9/120D;
		
		double delta = 0;
		
		while(true) {
			long now = System.nanoTime();
			delta += (now-lastTime)/ns;
			lastTime = now;
			while(delta >= 1) {
				delta--;
				tick();
			}
		}
	}
	
	public void tick() {
		int x = (int) ((mousePos.x-xo)/(double)scale);
		int y = (int) ((mousePos.y-yo)/(double)scale);
		int brushsize = 3;
		if(x > 56) {
			Arrays.setAll(pixels, j -> 0);
			Main.nn.feedForward(new double[pixels.length]);
		}
		if(mousePressed) {
			for(int iy = 0; iy < brushsize; iy++) {
				for(int ix = 0; ix < brushsize; ix++) {
					if((int)(ix*ix+iy*iy) > brushsize*brushsize) continue;
					int i = x+ix + (y+iy) * image.getWidth();	
					if(0 <= i && i < pixels.length)
					pixels[i] = 0xffffff;
				}
			}
			double[] dpix = new double[pixels.length];
			Arrays.setAll(dpix, j -> (double)pixels[j]/(double)0xffffff);
			Main.nn.feedForward(dpix);
		}
	}
	
	public void draw(Graphics g) {
		g.drawImage(image, xo, yo, image.getWidth()*scale, image.getHeight()*scale, null);
	}
	
	private Point mousePos = new Point();
	private boolean mousePressed = false;

	public void mouseClicked(MouseEvent e) {
		mousePos = new Point(e.getX(), e.getY());
	}

	public void mousePressed(MouseEvent e) {
		mousePos = new Point(e.getX(), e.getY());
		mousePressed = true;
	}

	public void mouseReleased(MouseEvent e) {
		mousePos = new Point(e.getX(), e.getY());
		mousePressed = false;
	}

	public void mouseEntered(MouseEvent e) {
		
	}

	public void mouseExited(MouseEvent e) {
		
	}

	public void mouseDragged(MouseEvent e) {
		mousePos = new Point(e.getX(), e.getY());
		mousePressed = true;
	}

	public void mouseMoved(MouseEvent e) {
		mousePos = new Point(e.getX(), e.getY());
	}
}