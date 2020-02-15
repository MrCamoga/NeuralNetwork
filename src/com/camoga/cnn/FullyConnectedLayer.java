package com.camoga.cnn;

import java.util.Arrays;
import java.util.Random;

import jdk.nashorn.api.tree.ArrayAccessTree;

public class FullyConnectedLayer {

	private double[][][] w;
	private double[][] a;
	private double[][] z;
//	private double[][] z;
	private double[][] b;
	private ActivationFunction[] f;
	private ErrorFunction error = ErrorFunction.CROSSENTROPY;
	
	private double[][][] djdw;
	private double[][] djdb;
	
	enum ErrorFunction {
		MEANSQUARED, CROSSENTROPY
	}
	
	enum ActivationFunction {
		SIGMOID, SOFTMAX, RELU
	}
	
	public FullyConnectedLayer(int...layersizes) {
		w = new double[layersizes.length-1][][];
		djdw = new double[w.length][][];
		for(int i = 0; i < w.length; i++) {
			w[i] = new double[layersizes[i]][layersizes[i+1]];
			djdw[i] = new double[layersizes[i]][layersizes[i+1]];
		}
		
		b = new double[layersizes.length-1][];
		djdb = new double[b.length][];
		for(int i = 0; i < b.length; i++) {
			b[i] = new double[layersizes[i+1]];
			djdb[i] = new double[layersizes[i+1]];
		}

		a = new double[layersizes.length][];
		for(int i = 0; i < a.length; i++) {
			a[i] = new double[layersizes[i]];
		}
		
		f = new ActivationFunction[a.length-1];
		for(int i = 0; i < f.length-1; i++) {
			f[i] = ActivationFunction.SIGMOID;
		}
		f[f.length-1] = ActivationFunction.SOFTMAX;
		System.out.println(Arrays.toString(layersizes));
		randinit();
	}
	
	public void randinit() {
		Random r = new Random();
		for(int i = 0; i < w.length; i++) {
			for(int m = 0; m < w[i].length; m++) {
				for(int n = 0; n < w[i][m].length; n++) {
					w[i][m][n] = (double) r.nextGaussian();
				}
			}
		}
		for(int i = 0; i < b.length; i++) {
			for(int m = 0; m < b[i].length; m++) {
				b[i][m] = (double) r.nextGaussian();
			}
		}
	}
	
	public double[] forward(double[][][] prev) {
		z = new double[a.length-1][];
		int index = 0;
		for(int n = 0; n < prev.length; n++) {
			for(int y = 0; y < prev[0].length; y++) {
				for(int x = 0; x < prev[0].length; x++) {
					this.a[0][index] = prev[n][y][x];
					index++;
				}
			}
		}
		
		for(int i = 0; i < z.length; i++) {
			z[i] = new double[a[i+1].length];
			for(int m = 0; m < a[i].length; m++) {
				for(int n = 0; n < a[i+1].length; n++) {
					z[i][n] += a[i][m]*w[i][m][n]+b[i][n];
				}
			}
			a[i+1] = activation(z[i], f[i]);
		}
		return a[a.length-1];
	}
	
	double cost = 0;
	public double[][][] backprop(ICLayer prev, double[] target) {
		cost += computeCost(target);
//		djdw = new double[w.length][][];
//		djdb = new double[b.length][];
//		for(int i = 0; i < w.length; i++) {
//			djdw[i] = new double[w[i].length][w[i][0].length];
//		}
//		for(int i = 0; i < b.length; i++) {
//			djdb[i] = new double[b[i].length];
//		}
//		
		double[] djdz = new double[target.length];
		for(int i = 0; i < djdz.length; i++) {
			djdz[i] = a[a.length-1][i]-target[i];
		}
		add(djdw[a.length-2],vecvecMul(a[a.length-2], djdz));
		add(djdb[a.length-2], djdz);
		
		for(int i = a.length-3; i >= 0; i--) {
			double[] djda = mulMatrix(w[i+1], djdz);
			double[] dadz = activationDerivative(a[i+1], f[i]);
			djdz = mul(djda,dadz);
//			System.out.println(Arrays.toString(djdz));
			add(djdw[i], vecvecMul(a[i], djdz));
			add(djdb[i], djdz);
		}
		
		double[] djdi = mulMatrix(w[0], djdz);
		double[][][] cnninput = new double[prev.depth()][prev.size()][prev.size()];
		int index = 0;
		for(int n = 0; n < prev.depth(); n++) {
			for(int y = 0; y < prev.size(); y++) {
				for(int x = 0; x < prev.size(); x++) {
					cnninput[n][y][x] = djdi[index];
					index++;
				}
			}
		}
		
		
		return cnninput;
	}
	
	private double[] mul(double[] a, double[] b) {
		double[] c = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			c[i] = a[i]*b[i];
		}
		return c;
	}
	
	private double[] mulMatrix(double[][] a, double[] z) {
		double[] result = new double[a.length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				result[i] += a[i][j] * z[j];
			}
		}
		return result;
	}
	
	private void add(double[] a, double[] b) {
		for(int i = 0; i < a.length; i++) {
			a[i] += b[i];
		}
	}
	
	private void add(double[][] a, double[][] b) {
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[i].length; j++) {
				a[i][j] += b[i][j];
			}
		}
	}
	
	private double[][] vecvecMul(double[] a, double[] djdz) {
		double[][] djdw = new double[a.length][djdz.length];
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < djdz.length; j++) {
				djdw[i][j] = a[i]*djdz[j];
			}
		}
		return djdw;
	}
	
	private double[] costDerivative(double[] target) {
		double[] djdy = new double[target.length];
		switch (error) {
		case MEANSQUARED:
			//TODO
			break;

		case CROSSENTROPY:
			break;
		}
		return djdy;
	}
	
	private double computeCost(double[] target) {
		double cost = 0;
		switch (error) {
		case MEANSQUARED:
			//TODO
			break;

		case CROSSENTROPY:
			for(int i = 0; i < target.length; i++) {
				cost -= target[i]*Math.log(a[a.length-1][i]);
			}
			break;
		}
		return cost;
	}
	
	private double[] activationDerivative(double[] a, ActivationFunction f) {
		double[] result = new double[a.length];
		switch (f) {
		case SIGMOID:
			for(int i = 0; i < result.length; i++) {
				result[i] = a[i]*(1-a[i]);
			}
			break;
		case RELU:
			for(int i = 0; i < result.length; i++) {
				result[i] = a[i] > 0 ? 1:0;
			}
		case SOFTMAX:
			break;
		}
		return result;
	}

	private double[] activation(double[] v, ActivationFunction f) {
		double[] a = new double[v.length];
		switch (f) {
		case SIGMOID:
			for(int i = 0; i < a.length; i++) {
				a[i] = (double) (1/(1+Math.exp(-v[i])));
			}
			break;
		case RELU:
			for(int i = 0; i < a.length; i++) {
				a[i] = Math.max(0, v[i]);
			}
			break;
		case SOFTMAX:
			double sum = 0;
			double max = Arrays.stream(v).max().getAsDouble();
			for(int i = 0; i < a.length; i++) {
				a[i] = (double) Math.exp(v[i]-max);
				sum += a[i];
			}
			for(int i = 0; i < a.length; i++) {
				a[i] /= sum;
			}
			break;
		}
		
		return a;
	}

	public double[] output() {
		return null;
	}

	public int depth() {
		return 0;
	}

	public int size() {
		return 0;
	}

	public void gradientDescent() {
		System.out.println(cost);
		for(int i = 0; i < w.length; i++) {
			for(int m = 0; m < w[i].length; m++) {
				for(int n = 0; n < w[i][m].length; n++) {
					w[i][m][n] -= ConvNet.learningrate*djdw[i][m][n];
					djdw[i][m][n] = 0;
				}
			}
		}
		for(int i = 0; i < b.length; i++) {
			for(int m = 0; m < b[i].length; m++) {
				b[i][m] -= ConvNet.learningrate*djdb[i][m];
				djdb[i][m] = 0;
			}
		}
		
		cost = 0;
		
	}
}