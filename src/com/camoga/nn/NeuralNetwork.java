package com.camoga.nn;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Random;

import org.omg.Messaging.SyncScopeHelper;

public class NeuralNetwork {

	public double[][] a;

	public double[][][] w;
	public double[][] b;
	public ActivationFunctions[] f;

	public static double lambda = 0.1;
	
	public int correct = 0;
	public int total = 0;

	/**
	 * After calling this constructor you should set the activation functions with setActivations();
	 * @param layerSizes
	 */
	public NeuralNetwork(int...layerSizes) {
		initNetwork(layerSizes);
		randomInit();
	}
	
	public void setActivations(int...id) {
		if(id.length != w.length) throw new RuntimeException("Length of activation functions is incorrect");
		this.f = new ActivationFunctions[id.length];
		for(int i = 0; i < id.length; i++) {
			for(ActivationFunctions f : ActivationFunctions.values()) {
				if(f.getId()==id[i]) {
					this.f[i] = f;
					break;
				}
				throw new RuntimeException("Could not find activation id");
			}
		}
	}
	
	public NeuralNetwork(BufferedInputStream bis) throws IOException { 
		int layers = bis.read();	//TODO unlimited num of layers
		byte[] bytes = new byte[4];
		int[] l = new int[layers];
		for(int n = 0; n < layers; n++) {
			bis.read(bytes);
			l[n] = ByteBuffer.wrap(bytes).getInt();
		}
		initNetwork(l);
		int[] activations = new int[layers-1];
		for(int n = 0; n < layers-1; n++) {
			activations[n] = bis.read();
		}
		setActivations(activations);	
		for(int n = 0; n < w.length; n++) {
			for(int k = 0; k < w[n].length; k++) {
				for(int j = 0; j < w[n][k].length; j++) {
					byte[] array = new byte[8];
					bis.read(array);
					double weight = ByteBuffer.wrap(array).getDouble();
					w[n][k][j] = weight;
				}
			}
		}
		
		for(int n = 0; n < b.length; n++) {
			for(int j = 0; j < b[n].length; j++) {
				byte[] array = new byte[8];
				bis.read(array);
				b[n][j] = ByteBuffer.wrap(array).getDouble();
			}
		}
	}

	public void randomInit() {
		Random r = new Random();
		for(int n = 0; n < w.length; n++) {
			for (int i = 0; i < w[n].length; i++)
				for (int j = 0; j < w[n][i].length; j++)
					w[n][i][j] = r.nextDouble()  - 0.5;
		}
	}

	public void initNetwork(int...l) {
		a = new double[l.length][];
		for(int j = 0; j < a.length; j++) {
			a[j] = new double[l[j]];
		}

		w = new double[a.length-1][][];
		for(int j = 0; j < w.length; j++) {
			w[j] = new double[l[j+1]][l[j]];
		}
		
		b = new double[w.length][];
		for(int j = 0; j < b.length; j++) {
			b[j] = new double[l[j+1]];
		}
		Arrays.stream(l).forEach(i -> System.out.print(i+"×"));
	}

  //Stochastic training
	public void train(double[][] input, double[][] target) {
		double[][][] dJdw = new double[w.length][][];
		double[][] dJdb = new double[b.length][];
		
		for(int n = 0; n < dJdw.length; n++) {
			dJdw[n] = new double[w[n][0].length][w[n].length];
			dJdb[n] = new double[a[n+1].length];
		}

		double cost = 0;
		for(int i = 0; i < input.length; i++) {
			double[][] z = feedForward(input[i], target[i]);
			
			//Backprop
			double[] dJdy = sub(a[a.length-1], target[i]);
			double[] dydzl = activationPrime(z[z.length-1],f[f.length-1]);
			double[] dJdzl = mul(dJdy, dydzl);
			dJdw[a.length-2] = add(dJdw[a.length-2],matrixMul(a[a.length-2], dJdzl));
			dJdb[a.length-2] = add(dJdb[a.length-2],dJdzl);
			
			for(int n = z.length-2; n > 0; n--) {
				double[] dJda = mulMatrix(w[n+1], dJdzl);
				double[] dadz = activationPrime(z[n],f[n]);
				double[] dJdz = mul(dJda, dadz);
				dJdw[n] = add(dJdw[n], matrixMul(a[n], dJdz));
				dJdb[n] = add(dJdb[n], dJdz);
			}
			
			cost += computeCost(a[a.length-1], target[i])/input.length;
			
			checkOutput(target[i]);
		}
		
		
		gradientDescent(dJdw, dJdb);
//		System.out.println("\ncost derivative:");
//		Arrays.stream(dJdy).forEach(i -> System.out.println(i));
		System.out.println("\nJ = " + cost);
//
//		System.err.println("Updating weights...");
		
	}
	
	public double[][] feedForward(double[] input, double[] target) {
		double[][] z = new double[a.length-1][];
		this.a[0] = input;
		for(int i = 0; i < z.length; i++) {
			z[i] = add(b[i],matrixVecMult(a[i], w[i]));
			a[i+1] = activation(z[i],f[i]);
		}
		checkOutput(target);
		return z;
	}
	
	public boolean checkOutput(double[] target) {
		total++;
		int biggest = 0;
		for(int j = 0; j < target.length; j++) {
			biggest = a[a.length-1][j] > a[a.length-1][biggest] ? j:biggest;
		}
		if(target[biggest] == 1) {
			correct++;
			return true; 
		}
		return false;
	}
	
	public void gradientDescent(double[][][] dJdw, double[][] dJdb) {
		for(int n = 0; n < w.length; n++) {
			for (int i = 0; i < w[n].length; i++) {
				for (int j = 0; j < w[n][i].length; j++) {
//					dw[n][i][j] = +lambda * dJdw[n][j][i];
					w[n][i][j] -= lambda * dJdw[n][j][i]; // TODO rotate matrix						
					
				}
			}
			for(int i = 0; i < b[n].length; i++) {
				b[n][i] += -lambda*dJdb[n][i];
			}
		}
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
	
	public double[] add(double a[], double[] b) {
		double[] result = new double[b.length];
		Arrays.setAll(result, i -> a[i]+b[i]);
		return result;
	}

	public double[] sub(double a, double[] b) {
		double[] result = new double[b.length];
		Arrays.setAll(result, i -> a - b[i]);
		return result;
	}

	public double[] activation(double[] layer, ActivationFunctions f) {
		double[] output = new double[layer.length];
		switch(f) {
		case SIGMOID:
			for (int i = 0; i < layer.length; i++) {
				output[i] = 1 / (1 + Math.exp(-layer[i]));
			}
			break;
		case TANH:
			for (int i = 0; i < layer.length; i++) {
				double ex = Math.exp(-2*layer[i]);
				output[i] = 2/(1+ex)-1;
			}
			break;
		}
		return output;
	}

	public double[] activationPrime(double[] layer, ActivationFunctions f) {
		double[] result = null;
		double[] s = activation(layer, f);
		switch(f) {
		case SIGMOID:
			result = mul(s, sub(1, s));
			break;
		case TANH:
			result = sub(1, mul(s, s));
			break;
		}
		return result;
	}
}
