package com.camoga.nn;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork implements Cloneable {

	public double[][] a;

	public double[][][] w;
	public double[][] b;
	public ActivationFunctions[] f;
	public CostFunctions COSTFUNCTION = CostFunctions.QUADRATIC;

	public static double lambda = 0.1;
	
	//TODO remove checking
	public int correct = 0;
	public int total = 0;
	
	private double maxCost = 0;
	public ArrayList<Double> costOverTime = new ArrayList<Double>();

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
		for(int i = 0; i < id.length;i++) {
			for(ActivationFunctions f : ActivationFunctions.values()) {
				if(f.getId()==id[i]) {
					this.f[i] = f;
					break;
				}
			}
			//TODO
//			throw new RuntimeException("Could not find activation id");
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
				for (int j = 0; j < w[n][i].length; j++) {
					w[n][i][j] = r.nextGaussian()*0.2;					
				}
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
//		Arrays.stream(l).forEach(i -> System.out.print(i+"×"));
	}

	/**
	 * 
	 * @param input
	 * @param target
	 * @return cost with respect to the input neurons, in case it's needed for further backprop. dJ/da0
	 */
	public double[] train(double[] input, double[] target) {
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
		double[] dydz = activationPrime(z[z.length-1],f[f.length-1]);
		double[] dJdz = mul(dJdy, dydz);
		dJdw[a.length-2] = add(dJdw[a.length-2],matrixMul(a[a.length-2], dJdz));
		dJdb[a.length-2] = add(dJdb[a.length-2],dJdz);
		
		for(int n = z.length-2; n >= 0; n--) {
			double[] dJda = mulMatrix(w[n+1], dJdz);
			double[] dadz = activationPrime(z[n],f[n]);
			dJdz = mul(dJda, dadz);
			dJdw[n] = matrixMul(a[n], dJdz);
			dJdb[n] = dJdz;
		}
		
		cost += computeCost(a[a.length-1], target);
		
		checkOutput(target);

		gradientDescent(dJdw, dJdb);
		
		System.out.println("\nJ = " + cost);
//		costOverTime.add(cost);
		return mulMatrix(w[0], dJdz);
	}
	
	public double[][] batchTrain(double[][] input, double[][] target) {
		double[][][] dJdw = new double[w.length][][];
		double[][] dJdb = new double[b.length][];
		
		for(int n = 0; n < dJdw.length; n++) {
			dJdw[n] = new double[w[n][0].length][w[n].length];
			dJdb[n] = new double[a[n+1].length];
		}
		
		double dJda1[][] = new double[input.length][a[0].length];

		double cost = 0;
		for(int i = 0; i < input.length; i++) {
			double[][] z = feedForward(input[i]);
			
			//Backprop
			double[] dJdy = costPrime(target[i]);
			double[] dydz = activationPrime(z[z.length-1],f[f.length-1]);
			double[] dJdz = mul(dJdy, dydz);
			dJdw[a.length-2] = add(dJdw[a.length-2],matrixMul(a[a.length-2], dJdz));
			dJdb[a.length-2] = add(dJdb[a.length-2],dJdz);
			
			for(int n = z.length-2; n >= 0; n--) {
				double[] dJda = mulMatrix(w[n+1], dJdz);
				double[] dadz = activationPrime(z[n],f[n]);
				dJdz = mul(dJda, dadz);
				dJdw[n] = add(dJdw[n], matrixMul(a[n], dJdz));
				dJdb[n] = add(dJdb[n], dJdz);
			}
			
			dJda1[i] = add(dJda1[i], mulMatrix(w[0], dJdz));
//			System.out.println(dJda1[i][0]);
			cost += computeCost(a[a.length-1], target[i]);
			
			checkOutput(target[i]);
		}
		
		
		gradientDescent(dJdw, dJdb);
//		System.out.println("\ncost derivative:");
//		Arrays.stream(dJdy).forEach(i -> System.out.println(i));
		System.out.println("\nJ = " + cost);
		if(cost > maxCost) maxCost = cost;
		costOverTime.add(cost);
//		System.err.println("Updating weights...");
		
		return dJda1;
	}
	
	/**
	 * returns neural network output
	 */
	public double[] feed(double[] input) {
		double[][] z = new double[a.length-1][];
		this.a[0] = input;
		for(int i = 0; i < z.length; i++) {
			z[i] = add(b[i],matrixVecMult(a[i], w[i]));
			a[i+1] = activation(z[i],f[i]);
		}
		return a[z.length];
	}
	
	public double[][] feedForward(double[] input) {
		double[][] z = new double[a.length-1][];
		this.a[0] = input;
		for(int i = 0; i < z.length; i++) {
			z[i] = add(b[i],matrixVecMult(a[i], w[i]));
			a[i+1] = activation(z[i],f[i]);
		}
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
					w[n][i][j] -= lambda * dJdw[n][j][i]; // TODO rotate matrix						
					
				}
			}
			for(int i = 0; i < b[n].length; i++) {
				b[n][i] -= lambda*dJdb[n][i];
			}
		}
	}

	public double computeCost(double[] target) {
		return computeCost(a[a.length-1], target);
	}
	
	public double computeCost(double[] output, double[] target) {
		double result = 0;
		for (int i = 0; i < output.length; i++) {
			switch(COSTFUNCTION) {
				case QUADRATIC:
					result += 0.5 * Math.pow((output[i] - target[i]), 2);
					break;
				case CROSSENTROPY:
					result += -(target[i]*Math.log(output[i])+(1-target[i])*Math.log(1-output[i]));
					break;
			}
		}
		return result/(double)output.length;
	}
	
	public double[] costPrime(double[] target) {
		double[] result = null;
		switch (COSTFUNCTION) {
			case QUADRATIC:
				result = sub(a[a.length-1],target);
				break;

			case CROSSENTROPY:
				result = new double[target.length];
				for(int i = 0; i < result.length; i++) {
					result[i] = (a[a.length-1][i] - target[i])/((1-a[a.length-1][i])*a[a.length-1][i]);
				}
				break;
		}
		
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
//		System.out.println(a.length+"x"+a[0].length);
//		System.out.println(z.length);
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
	
	public NeuralNetwork mutate(double mutationrate, double mutation) {
		for(int i = 0; i < w.length; i++) {
			for(int j = 0; j < w[i].length; j++) {
				for(int k = 0; k < w[i][j].length; k++) {
					if(Math.random() < mutationrate) w[i][j][k] += (Math.random()-0.5)*1.5*mutation;
				}
			}
		}
		return this;
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
		case RELU:
			for (int i = 0; i < layer.length; i++) {
				output[i] = Math.max(0, layer[i]);				
			}
		}
		return output;
	}

	public double[] activationPrime(double[] layer, ActivationFunctions f) {
		double[] result = null;
		switch(f) {
		case SIGMOID:
			double[] s = activation(layer, f);
			result = mul(s, sub(1, s));
			break;
		case TANH:
			double[] s2 = activation(layer, f);
			result = sub(1, mul(s2, s2));
			break;
		case RELU:
			result = new double[layer.length];
			for(int i = 0; i < layer.length; i++) {
				result[i] = layer[i] > 0 ? 1:0;
			}
		}
		return result;
	}
	
	public NeuralNetwork clone() {
		try {
			return (NeuralNetwork) super.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public void renderCostPlot(Graphics g, int xo, int yo, int width, int height, int color) {
		
		//Points
		int x, y;
		int xlast = 0, ylast = 0;
		g.setColor(new Color(color));
		int maxpoints = 200;
		int start = (int)Math.max(0, costOverTime.size()-maxpoints);
		double maxCost = 0;
		double minCost = Double.MAX_VALUE;
		int size = (int)Math.min(costOverTime.size(), maxpoints);
		for(int i = 0; i < size; i+=1) {
			int index = i + start;
			x = (int) (xo + width*i/size);
			if(maxCost < costOverTime.get(index)) maxCost = costOverTime.get(index);
			if(minCost > costOverTime.get(index)) minCost = costOverTime.get(index);
			y = (int) (yo + height*(1-costOverTime.get(index)/(maxCost)));
			
			if(i > 0) g.drawLine(xlast, ylast, x, y);
			g.fillOval(x-1, y-1, 2, 2);
			xlast = x;
			ylast = y;
		}
		
		//Plot
		g.setColor(Color.BLACK);
		//Axes
		g.drawLine(xo, yo, xo, yo+height);
		g.drawLine(xo, yo+height, xo+width, yo+height);
		g.drawString(maxCost+"", xo, yo);
		
		int num = (int)(10*Math.ceil(costOverTime.size()/100));
		if(num == 0) num = 2;
		
		for(int m = 0; m <= costOverTime.size()/num; m++) {
			int xm = (int) (xo+m*num/(double)costOverTime.size()*width);
//			System.out.println(costOverTime.size());
//			System.out.println(xm);
			g.drawLine(xm, yo+height-10, xm, yo+height+10);
			g.drawString(start+m*num+"", xm, yo+height+20);
		}
		
		//Current
		g.drawLine(xo, ylast, xo+width, ylast);
		if(costOverTime.size() > 0)g.drawString(""+costOverTime.get(costOverTime.size()-1), xo, ylast);
	}

	public void renderNN(Graphics g, int xo, int yo, int width, int height) {
		int diameter = 15;
		int yspacing = 7;
		int xspacing = 100;
		int maxneurons = 10;
		
		int[] yoa = new int[a.length];
		for(int n = 0; n < yoa.length; n++) {
			yoa[n] = a[n].length > maxneurons ? (height-maxneurons*(diameter+yspacing))/2:(height-a[n].length*(diameter+yspacing))/2;
		}
		g.setFont(new Font("Arial", Font.BOLD, 9));
		FontMetrics fontm = g.getFontMetrics();
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
				String o = new DecimalFormat("#.##").format(activation)+"";
				g.drawString(o, 
						xspacing*n+xo+(diameter-fontm.stringWidth(o))/2, 
						yo+y+(diameter-fontm.getHeight())/2+fontm.getAscent());
			}
		}
		for(int n = 0; n < b.length; n++) {
			g.setColor(Color.white);
			g.fillOval(xspacing*n+xo, yo+height-yoa[n], diameter, diameter);
		}
	}
	
	/**
	 * Save all neural network
	 * @param path
	 */
	public void save(String path) {
		saveLayers(0, a.length, path);
	}
	
	/**
	 * Save some layers
	 * @param layerStart
	 * @param layerEnd
	 * @param path
	 */
	public void saveLayers(int layerStart, int layerEnd, String path) {
		try {
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(path));
			bos.write(layerEnd-layerStart);
			//NN Layers
			for(int n = layerStart; n < layerEnd; n++) {
				byte[] bytes = new byte[4];
				ByteBuffer.wrap(bytes).putInt(a[n].length);
				bos.write(bytes);
			}
			for(int n = layerStart; n < layerEnd-1; n++) {
				bos.write(f[n].getId());
			}
			//NN weights
			for(int i = layerStart; i < layerEnd-1; i++) {
				for(int j = 0; j < w[i].length; j++) {
					for(int k = 0; k < w[i][j].length; k++) {
						byte[] bytes = new byte[8];
						ByteBuffer.wrap(bytes).putDouble(w[i][j][k]);
						bos.write(bytes);
//						System.err.println(Arrays.toString(bytes));
					}
				}
			}
			//NN Biases
			for(int i = layerStart; i < layerEnd-1; i++) {
				for(int j = 0; j < b[i].length; j++) {
					byte[] bytes = new byte[8];
					ByteBuffer.wrap(bytes).putDouble(b[i][j]);
					bos.write(bytes);
				}
			}
			System.err.println("Saved");
			bos.flush();
			bos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void saveCost(String path) {
		try {
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(path));
		
			for(double c : costOverTime) {
				byte[] bytes = new byte[8];
				ByteBuffer.wrap(bytes).putDouble(c);
				bos.write(bytes);
			}
			System.err.println("Saved!");
			bos.flush();
			bos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}