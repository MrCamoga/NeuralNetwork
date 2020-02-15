package com.camoga.cnn;

import java.awt.Graphics;

public class ReLULayer implements ICLayer {

	private double[][][] relu;
	
	public ReLULayer(ICLayer prev) {
		relu = new double[prev.depth()][prev.size()][prev.size()];
	}
	
	public double[][][] forward(double[][][] prev) {
		relu = new double[depth()][size()][size()];
		for(int n = 0; n < depth(); n++) {
			for(int y = 0; y < size(); y++) {
				for(int x = 0; x < size(); x++) {
					relu[n][y][x] = prev[n][y][x] < 0 ? 0:prev[n][y][x];
				}
			}
		}
		return relu;
	}
	
	double[][][] djdr;
	
	public double[][][] backprop(ICLayer prev, double[][][] cost) {
		djdr = new double[depth()][size()][size()];
		for(int n = 0; n < depth(); n++) {
			for(int y = 0; y < size(); y++) {
				for(int x = 0; x < size(); x++) {
					djdr[n][y][x] = relu[n][y][x] == 0 ? 0:cost[n][y][x];
				}
			}
		}
		return djdr;
	}
	
	public double[][][] output() {
		return relu;
	}

	public int depth() {
		return relu.length;
	}

	public int size() {
		return relu[0].length;
	}

	public int render(Graphics g, int x) {
		return 120;
	}
}