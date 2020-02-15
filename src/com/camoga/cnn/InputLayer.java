package com.camoga.cnn;

import java.awt.Graphics;

public class InputLayer implements ICLayer {

	private double[][][] input;
	
	public InputLayer(int channels, int size) {
		input = new double[channels][size][size];
	}

	public double[][][] forward(double[][][] input) {
		this.input = input;
		return input;
	}
	
	public double[][][] backprop(ICLayer layer, double[][][] prev) {
		return null;
	}

	public int depth() {
		return input.length;
	}
	
	public int size() {
		return input[0].length;
	}

	public double[][][] output() {
		return input;
	}

	public int render(Graphics g, int x) {
		for(int i = 0; i < depth(); i++) {
			g.drawImage(Window.createImage(input[i]), x, i*120, null);
		}
		
		g.drawImage(Window.createImage(input), x, 360, null);
		return 120;
	}
}