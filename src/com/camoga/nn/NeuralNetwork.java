package com.camoga.nn;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork implements Cloneable{

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
	 * @param activations {@link ActivationFunctions}
	 */
	public NeuralNetwork(int[] layerSizes, int[] activations) {
		initNetwork(layerSizes);
		randomInit();
		setActivations(activations);
	}
	
	/**
	 * @param id look {@link ActivationFunctions}
	 */
	public void setActivations(int...id) {
		if(id.length != w.length) throw new RuntimeException("Length of activation functions is incorrect");
		this.f = new ActivationFunctions[id.length];
		for(int i = 0; i < id.length; i++) {
			for(ActivationFunctions f : ActivationFunctions.values()) {
				if(f.getId() == id[i]) {
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

	public void train(double[] input, double[] target) {
		double[][][] dJdw = new double[w.length][][];
		double[][] dJdb = new double[b.length][];
		
		for(int n = 0; n < dJdw.length; n++) {
			dJdw[n] = new double[w[n][0].length][w[n].length];
			dJdb[n] = new double[a[n+1].length];
		}

		double cost = 0;
		
		double[][] z = feedForward(input);
		
		//Backprop
		double[] dJdy = sub(a[a.length-1], target);
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
		
		cost += computeCost(a[a.length-1], target)/input.length;
		
		checkOutput(target);

		gradientDescent(dJdw, dJdb);
		
		System.out.println("\nJ = " + cost);
	}
	
	public void batchTrain(double[][] input, double[][] target) {
		double[][][] dJdw = new double[w.length][][];
		double[][] dJdb = new double[b.length][];
		
		for(int n = 0; n < dJdw.length; n++) {
			dJdw[n] = new double[w[n][0].length][w[n].length];
			dJdb[n] = new double[a[n+1].length];
		}

		double cost = 0;
		for(int i = 0; i < input.length; i++) {
			double[][] z = feedForward(input[i]);
			
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
	
	/**
	 * @return neural network output
	 */
	public double[] feed(double[] input) {
		double[][] z = new double[a.length-1][];
		this.a[0] = input;
		for(int i = 0; i < z.length; i++) {
			z[i] = add(b[i],matrixVecMult(a[i], w[i]));
			a[i+1] = activation(z[i],f[i]);
		}
//		double[] z1 = add(b[0],matrixVecMult(a[0], w[0]));
//		a[1] = activation(z1,f); // hidden layer
//		double[] z2 = add(b[1],matrixVecMult(a[1], w[1]));
//		a[2] = activation(z2,f); // hidden layer
//		double[] z3 = add(b[2],matrixVecMult(a[2], w[2]));
//		a[3] = activation(z3,f); // output layer
//		System.out.println(z.length+" x " + z[0].length);
		return a[z.length];
	}
	
	public double[][] feedForward(double[] input) {
		double[][] z = new double[a.length-1][];
		this.a[0] = input;
		for(int i = 0; i < z.length; i++) {
			z[i] = add(b[i],matrixVecMult(a[i], w[i]));
			a[i+1] = activation(z[i],f[i]);
		}
//		double[] z1 = add(b[0],matrixVecMult(a[0], w[0]));
//		a[1] = activation(z1,f); // hidden layer
//		double[] z2 = add(b[1],matrixVecMult(a[1], w[1]));
//		a[2] = activation(z2,f); // hidden layer
//		double[] z3 = add(b[2],matrixVecMult(a[2], w[2]));
//		a[3] = activation(z3,f); // output layer
//		System.out.println(z.length+" x " + z[0].length);
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

	//TODO 
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
	
	public void mutate(double mutationrate, double mutation) {
		for(double[][] i : w) {
			for(double[] j : i) {
				for(double k : j) {
					if(Math.random() < mutationrate) k += k*mutation;
				}
			}
		}
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

	/**
	 * 
	 * @param layer
	 * @param f
	 * @return f'(z)
	 */
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
	
	/**
	 * 
	 */
	public NeuralNetwork clone() {
		try {
			return (NeuralNetwork) super.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public void render(Graphics g, int xo, int yo, int height, int width) {
		int diameter = 20;
		int yspacing = 10;
		int xspacing = 100;
		int maxneurons = 12;
		
		int[] yoa = new int[a.length];
		for(int n = 0; n < yoa.length; n++) {
			yoa[n] = a[n].length > maxneurons ? (height-maxneurons*(diameter+yspacing))/2:(height-a[n].length*(diameter+yspacing))/2;
		}
		
		g.setFont(new Font("Arial", Font.BOLD, 9));
		g.setColor(new Color(0xb0, 0xc0, 0x20));
		g.fillRect(xo-10, yo-10, width, height);
		
		//Weights
		for(int n = 0; n < w.length; n++) {
			for(int i = 0; i < Math.min(maxneurons,w[n][0].length); i++) {
				for(int j = 0; j < Math.min(maxneurons,w[n].length); j++) {
					double weight = w[n][j][i];
					g.setColor(weight>0 ? new Color(0, 0, 0xff, (int) Math.min(0xff,0xff*weight)):new Color(0xff, 0, 0, (int) (Math.min(0xff,0xff*-weight))));
					g.drawLine(xspacing*n+xo+diameter/2, yo+yoa[n]+i*(diameter+yspacing)+diameter/2, xo+xspacing*(n+1)+diameter/2, yo+ yoa[n+1]+j*(yspacing+diameter)+diameter/2);
				}
			}
		}
		//Biases
		for(int n = 0; n < b.length; n++) {
			for(int i = 0; i < Math.min(maxneurons,b[n].length); i++) {
				double wb = b[n][i];
				g.setColor(wb>0 ? new Color(0, 0xff, 0, (int) Math.min(0xff,0xff*wb)):new Color(0xff, 0x70, 0, (int) (Math.min(0xff,0xff*-wb))));
				g.drawLine(xspacing*n+xo+diameter/2, yo+height-yoa[n]+diameter/2, xspacing*(n+1)+xo+diameter/2, yo+yoa[n+1]+i*(diameter+yspacing)+diameter/2);
			}				
		}
		//Neurons
		for(int n = 0; n < a.length; n++) {
			for(int i = 0; i < Math.min(maxneurons,a[n].length); i++) {
				double activation = a[n][i];
				int col = (int) Math.min(0xff,Math.abs(0xff*activation));
				g.setColor(new Color(col,col,col));
				int y = yoa[n]+i*(diameter+yspacing);
				g.fillOval(xspacing*n+xo, y+yo, diameter, diameter);
				g.setColor(activation>0.8 ? Color.black:Color.white);
				g.drawString(new DecimalFormat("#.##").format(activation)+"", xspacing*n+xo+diameter/4-3, yo+y+diameter/4*3);
			}
		}
		for(int n = 0; n < b.length; n++) {
			g.setColor(Color.white);
			g.fillOval(xspacing*n+xo, yo+height-yoa[n], diameter, diameter);
		}
	}
}