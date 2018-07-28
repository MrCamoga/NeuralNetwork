package com.camoga.examples.numberrecognition;

import java.io.BufferedInputStream;
import java.io.IOException;

import com.camoga.nn.NeuralNetwork;

public class Main {
	
	private int batchsize = 120;
	public static NeuralNetwork nn;
	public Window window;
	
	public static boolean testing = false;
	
	public Main(String path) {
		initNN(path);
		nn.lambda = 0.005;
		trainNN();
//		testNN("/train-images.idx3-ubyte", "/train-labels.idx1-ubyte");
//		testNN("/t10k-images.idx3-ubyte", "/t10k-labels.idx1-ubyte");
	}
	
	public static void main(String[] args) {
//		new Main("/neuralnet/784x100x100x10.gaconvi");
		new Main(null);
	}
	
	public void initNN(String path) {
		if(path == null) {
			nn = new NeuralNetwork(784, 20,20, 10);
			nn.setActivations(new int[3]);
		}
		else try {
			nn = new NeuralNetwork(new BufferedInputStream(getClass().getResourceAsStream(path)));
//			nn.setActivations(new int[]{0,0,0});
		} catch (IOException e) {
			e.printStackTrace();
		}
		window = new Window();
	}
	
	public void trainNN() {
		double[][] pixels = new double[batchsize][28*28];
		double[][] targets;
		
		
		try {
			for(int a = 0; a < 10000; a++) {
				BufferedInputStream imagesBuffer = new BufferedInputStream(getClass().getResourceAsStream("/train-images.idx3-ubyte"));
				BufferedInputStream labelsBuffer = new BufferedInputStream(getClass().getResourceAsStream("/train-labels.idx1-ubyte"));
				imagesBuffer.skip(16);
				labelsBuffer.skip(8);
				for(int batch = 0; batch < 60000/batchsize; batch++) {
//					batchsize = 60;
					targets = new double[batchsize][10];
					for(int i = 0; i < batchsize; i++) {
						for(int j = 0; j < pixels[i].length; j++) {
							pixels[i][j] = imagesBuffer.read()/255.0D;
						}
						int number = labelsBuffer.read();
						targets[i][number] = 1;
					}				
					feedNN(pixels, targets);
				}
			testNN("/t10k-images.idx3-ubyte", "/t10k-labels.idx1-ubyte");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void testNN(String images, String labels) {
		BufferedInputStream testImages = new BufferedInputStream(getClass().getResourceAsStream(images));
		BufferedInputStream testLabels = new BufferedInputStream(getClass().getResourceAsStream(labels));
		nn.correct = 0;
		nn.total = 0;
		testing = true;
		double cost = 0;
		try {
			testImages.skip(16);
			testLabels.skip(8);
			double[] pixels = new double[28*28];
			double[] target;
			for(int i = 0; i < 10000; i++) {
				target = new double[10];
				
				for(int j = 0; j < pixels.length; j++) {
					pixels[j] = testImages.read()/255.0D;
				}
				int number = testLabels.read();
				target[number] = 1;
				
				nn.feedForward(pixels);
				nn.checkOutput(target);
				cost += nn.computeCost(target)/10000;
//				
//				Thread.sleep(1200);
			}
			System.out.println("Jtest = " + cost);
		} catch (Exception e) {
			e.printStackTrace();
		}
		testing = false;
	}
	
	public void feedNN(double[][] images, double[][] targets) {
		nn.batchTrain(images, targets);
	}
	
}