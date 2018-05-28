package com.camoga.nn;

import java.util.Arrays;
import java.util.Random;

public class CopyOfNeuralNetwork {

	public double[] a0;
	public double[] a1;
	public double[] a2;
	public double[] a3;

	public double[][] w0;
	public double[][] w1;
	public double[][] w2;

	private static final double lambda = 0.1;

	// private double[] secondLayer = new double[10];
	public CopyOfNeuralNetwork(int inputSize, int hiddenLayerSize, int hiddenLayerSize2, int outputSize) {
		initNetwork(inputSize, hiddenLayerSize, hiddenLayerSize2, outputSize);
		randomInit();
		for(int y = 0; y < 100000; y++) {
			Random r = new Random();
			double[][] input = new double[100][1];
			double[][] target = new double[100][1];
			for(int i = 0; i < input.length; i++) {
				input[i][0] = r.nextDouble()*Math.PI/2;
				target[i][0] = Math.sin(input[i][0]);
			}
			propagation(input, target);
			System.out.println(y);
			try {
				Thread.sleep(16);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	public static void main(String[] args) {
		new CopyOfNeuralNetwork(1, 12, 12, 1);
	}

	public void initNetwork(int i, int h, int h2, int o) {
		a0 = new double[i];
		a1 = new double[h];
		a2 = new double[h2];
		a3 = new double[o];

		w0 = new double[h][i];
		w1 = new double[h2][h];
		w2 = new double[o][h2];
	}

	public void propagation(double[][] input, double[][] target) {
		double[][] dJdw2 = new double[w2[0].length][w2.length];
		double[][] dJdw1 = new double[w1[0].length][w1.length];
		double[][] dJdw0 = new double[w0[0].length][w0.length];
		double cost = 0;
		for(int i = 0; i < a0.length; i++) {
			this.a0 = input[i];
			double[] z1 = matrixVecMult(a0, w0);
			a1 = sigmoid(z1); // hidden layer
			double[] z2 = matrixVecMult(a1, w1);
			a2 = sigmoid(z2); // hidden layer
			double[] z3 = matrixVecMult(a2, w2);
			a3 = sigmoid(z3); // output layer

			// Backprop
			double[] dJdy = sub(a3, target[i]);
			double[] dydz3 = sigmoidPrime(z3);
			double[] dJdz3 = mul(dJdy, dydz3);
			dJdw2 = add(dJdw2,matrixMul(a2, dJdz3));

			double[] dJda2 = mulMatrix(w2, dJdz3);
			double[] da2dz2 = sigmoidPrime(z2);
			double[] dJdz2 = mul(dJda2, da2dz2);
			dJdw1 = add(dJdw1,matrixMul(a1, dJdz2));

			double[] dJda1 = mulMatrix(w1, dJdz2);
			double[] da1dz1 = sigmoidPrime(z1);
			double[] dJdz1 = mul(dJda1, da1dz1);
			dJdw0 = add(dJdw0,matrixMul(a0, dJdz1));
			cost += computeCost(a3, target[i])/a0.length;
		}
		

//		System.out.println("\ncost derivative:");
//		Arrays.stream(dJdy).forEach(i -> System.out.println(i));
//
		System.out.println("\nJ = " + cost);
//		System.err.println("Updating weights...");
		for (int i = 0; i < w0.length; i++) {
			for (int j = 0; j < w0[i].length; j++) {
				w0[i][j] += -lambda * dJdw0[j][i];
			}
		}
		for (int i = 0; i < w1.length; i++) {
			for (int j = 0; j < w1[i].length; j++) {
				w1[i][j] += -lambda * dJdw1[j][i];
			}
		}
		for (int i = 0; i < w2.length; i++) {
			for (int j = 0; j < w2[i].length; j++) {
				w2[i][j] += -lambda * dJdw2[j][i];
			}
		}
	}

	public void backpropagation() {

	}

	public double computeCost(double[] output, double[] target) {
		double result = 0;
		for (int i = 0; i < output.length; i++)
			result += 0.5 * Math.pow((output[i] - target[i]), 2);
		return result;
	}

	public double[] matrixVecMult(double[] a, double[][] w) {
		double[] result = new double[w.length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < a.length; j++) {
				result[i] += a[j] * w[i][j];
			}
		}
		return result;
	}

	public double[][] matrixMul(double[] a, double[] z) {
		double[][] result = new double[a.length][z.length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = a[i] * z[j];
			}
		}
		return result;
	}

	public double[] mulMatrix(double[][] a, double[] z) {

		double[] result = new double[a[0].length];
		for (int i = 0; i < z.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				result[j] += a[i][j] * z[i];
			}
		}
		return result;
	}

	public double[] mul(double[] a, double[] b) {
		double[] result = new double[a.length];
		Arrays.setAll(result, i -> a[i] * b[i]);
		return result;
	}

	public double[][] mul(double[] a, double[][] b) {
		double[][] result = new double[b.length][b[0].length];
		Arrays.setAll(result, i -> mul(a, b[i]));
		return result;
	}

	public double[] sub(double[] a, double[] b) {
		double[] result = new double[b.length];
		Arrays.setAll(result, i -> a[i] - b[i]);
		return result;
	}
	
	public double[][] add(double[][] a, double[][] b) {
		double[][] result = new double[b.length][b[0].length];
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result[i].length; j++) {
				result[i][j] += a[i][j]+b[i][j];
			}
		}
		return result;
	}

	public double[] sub(double a, double[] b) {
		double[] result = new double[b.length];
		Arrays.setAll(result, i -> a - b[i]);
		return result;
	}

	public void randomInit() {
		Random r = new Random();
		for (int i = 0; i < w0.length; i++)
			for (int j = 0; j < w0[i].length; j++)
				w0[i][j] = r.nextDouble() * 2 - 1;
		for (int i = 0; i < w1.length; i++)
			for (int j = 0; j < w1[i].length; j++)
				w1[i][j] = r.nextDouble() * 2 - 1;
		for (int i = 0; i < w2.length; i++)
			for (int j = 0; j < w2[i].length; j++)
				w2[i][j] = r.nextDouble() * 2 - 1;
	}

	public double[] sigmoid(double[] layer) {
		double[] output = new double[layer.length];
		for (int i = 0; i < layer.length; i++) {
			output[i] = 1 / (1 + Math.exp(-layer[i]));
		}
		return output;
	}

	public double[] sigmoidPrime(double[] layer) {
		double[] s = sigmoid(layer);
		return mul(s, sub(1, s));
	}
}