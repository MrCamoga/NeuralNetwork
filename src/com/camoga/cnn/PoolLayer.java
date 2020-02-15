package com.camoga.cnn;

import java.awt.Graphics;

public class PoolLayer implements ICLayer {

	private double[][][] sample;
	
	private double[][][] maxmask;
	
	public PoolLayer(ICLayer prev) {
		sample = new double[prev.depth()][(prev.size()+1)/2][(prev.size()+1)/2];
	}
	
	public double[][][] forward(double[][][] prev) {
		sample = new double[depth()][size()][size()];
		maxmask = new double[prev.length][prev[0].length][prev[0].length];
		for(int n = 0; n < depth(); n++) {
			for(int y = 0; y < size(); y++) {
				for(int x = 0; x < size(); x++) {
					int ym = 2*y, xm = 2*x;
					double max = prev[n][ym][xm];
					for(int yi = 0; yi < 2; yi++) {
						int yo = 2*y+yi;
						if(yo >= prev[n].length) break;
						for(int xi = 0; xi < 2; xi++) {
							int xo = 2*x+xi;
							if(xo >= prev[n].length) break;
							double value = prev[n][yo][xo]; //TODO odd size images nullpointerexception
							if(value > max) {
								max = value;
								ym = yo;
								xm = xo;
							}
						}
					}
					sample[n][y][x] = max;
					maxmask[n][ym][xm] = 1;
				}
			}
		}
		return sample;
	}
	 
	
	public double[][][] backprop(ICLayer prev, double[][][] cost) {
		double[][][] dJds = new double[prev.depth()][prev.size()][prev.size()];
//		System.out.println(dJds.length+","+dJds[0].length+","+dJds[0][0].length);
		for(int n = 0; n < cost.length; n++) {
			for(int y = 0; y < cost[n].length; y++) {
				for(int x = 0; x < cost[n][y].length; x++) {
					if(maxmask[n][y][x] == 0) continue;
					dJds[n][y][x] = cost[n][y][x];
				}
			}
		}
		
		return dJds;
	}
	
	public double[][][] output() {
		return sample;
	}

	public int depth() {
		return sample.length;
	}

	public int size() {
		return sample[0].length;
	}

	public int render(Graphics g, int x) {
		return 120;
	}
}