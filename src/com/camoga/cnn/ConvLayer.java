package com.camoga.cnn;

import java.util.Random;

public class ConvLayer implements ICLayer {

	enum ConvType {
		VALID, SAME, FULL
	}
	
	double[][][][] kernel;
	
	private double[][][] conv;
	private int stride, padding;
	private int offset;
	private int kernelsize;
	private int inputsize;
	
	public double[][][][] djdk;
	public double[][][] djdi;
	
	/**
	 * 
	 * @param prev
	 * @param outputdepth
	 * @param kernelsize
	 * @param padding "valid", "same" or "full"
	 * @param stride
	 */
	public ConvLayer(ICLayer prev, int outputdepth, int kernelsize, String type, int stride) {
		if(kernelsize%2==0) throw new RuntimeException("Even size kernel");
		
		if(type.equals("valid")) padding = 0;
		else if(type.equals("same")) padding = kernelsize/2;
		else if(type.equals("full")) padding = kernelsize-1; //TODO is correct?
		else throw new RuntimeException("invalid convolution type");
		
		this.kernel = new double[prev.depth()][outputdepth][kernelsize][kernelsize];
		int size = (prev.size()+2*padding-kernelsize)/stride+1;
		this.conv = new double[outputdepth][size][size];
		
		this.stride = stride;
		this.kernelsize = kernelsize;
		this.offset = kernelsize/2 - padding;
		this.inputsize = prev.size();
		initKernels();
		System.out.println("pad: " + padding);
		System.out.println("off: " + offset);
		System.out.println("size: " + size);
		djdk = new double[kernel.length][outputdepth][kernelsize][kernelsize];
		djdi = new double[kernel.length][inputsize][inputsize];
	}
	
	public double[][][] forward(double[][][] input) {
		conv = new double[depth()][size()][size()];
		int kernelSize = (kernelsize-1)/2;
		for(int m = 0; m < kernel.length; m++) {
			for(int n = 0; n < kernel[m].length; n++) {
				for(int y = offset, yo = 0; y < size()+offset; y+=stride, yo++) {
					for(int x = offset, xo = 0; x < size()+offset; x+=stride, xo++) {
						for(int ky = 0; ky < kernelsize; ky++) {
							int yi = y + ky - kernelSize;
							if(yi < 0) continue;
							if(yi >= input[0].length) continue;
							for(int kx = 0; kx < kernelsize; kx++) {
								int xi = x + kx-kernelSize;
								if(xi < 0) continue;
								if(xi >= input[0].length) continue;
								conv[n][yo][xo] += input[m][yi][xi]*kernel[m][n][ky][kx];
							}
						}
					}
				}
			}
		}
		return conv;
	}
	
	private void initKernels() {
		Random r = new Random(20);
		for(int m = 0; m < kernel.length; m++) {
			for(int n = 0; n < kernel[0].length; n++) {
				for(int y = 0; y < kernelsize; y++) {
					for(int x = 0; x < kernelsize; x++) {
						kernel[m][n][y][x] = (double) r.nextGaussian();
					}
				}
			}
		}
	}
	
	public int depth() {
		return conv.length;
	}
	
	public int size() {
		return conv[0].length;
	}

	
	public double[][][] backprop(ICLayer prev, double[][][] cost) {
		int kernelSize = kernelsize/2;
//		djdk = new double[kernel.length][depth()][kernelsize][kernelsize];
//		djdi = new double[kernel.length][inputsize][inputsize];
		for(int m = 0; m < kernel.length; m++) {
			for(int n = 0; n < kernel[0].length; n++) {
				for(int y = 0; y < kernelsize; y++) {
					for(int x = 0; x < kernelsize; x++) {
						for(int oy = 0; oy < cost[0].length; oy++) {
							int yi = y+oy-padding;
							if(yi < 0) continue;
							if(yi >= inputsize) break;
							for(int ox = 0; ox < cost[0].length; ox++) {
								int xi = x+ox-padding;
								if(xi < 0) continue;
								if(xi >= inputsize) break;
								djdk[m][n][y][x] += prev.output()[m][yi][xi]*cost[n][oy][ox];
							}
						}
					}
				}				
			}
		}
		for(int m = 0; m < kernel.length; m++) {
			for(int n = 0; n < kernel[0].length; n++) {
				for(int y = 0; y < djdi[0].length; y++) {
					for(int x = 0; x < djdi[0].length; x++) {
						for(int ky = 0; ky < kernelsize; ky++) {
							int yo = y - offset + ky - kernelSize;
							if(yo < 0) continue;
							if(yo >= size()) break;
							for(int kx = 0; kx < kernelsize; kx++) {
								int xo = x - offset + kx - kernelSize;
								if(xo < 0) continue;
								if(xo >= size()) break;
								djdi[m][y][x] += cost[n][yo][xo]*kernel[m][n][kernelsize-ky-1][kernelsize-kx-1];	
							}
						}
					}
				}
			}
		}
		
		return djdi;
	}
	
//	public double[][][] debugdjdi(double[][][] input) {
//		double[][][] djdi = new double[kernel.][prev.size()][prev.size()];
//		
//		double cost = computecost(input);
//		
//		for(int y = 0; y < input[0].length; y++) {
//			for(int x = 0; x < input[0].length; x++) {
//				input[0][y][x] += 0.00001;
//				djdi[0][y][x] = (computecost(input)-cost)/0.00001f;
//				input[0][y][x] -= 0.00001;
//			}
//		}
//		return djdi;
//	}
	
	public double computecost(double[][][] input) {
		double[][][] result = forward(input);
		double cost = 0;
		for(int y = 0; y < result[0].length; y++) {
			for(int x = 0; x < result[0].length; x++) {
				cost += result[0][y][x]*result[0][y][x];
			}
		}
		return cost;
	}

	public double[][][] output() {
		return conv;
	}
}