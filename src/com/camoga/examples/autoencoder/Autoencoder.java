package com.camoga.examples.autoencoder;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.util.Random;

import org.apache.tools.ant.DirectoryScanner;

import com.camoga.nn.CostFunctions;
import com.camoga.nn.NeuralNetwork;
import com.camoga.utils.LoadImage;

public class Autoencoder {

	private int batchsize = 16;
	public NeuralNetwork nn;
	public Window window;
	
	public static int WIDTH;
	public static int HEIGHT;
	
	public static final int numOfFeatures = 80;
	
	public double[][] IMAGES;
	public double[][] VALIDATIONIMAGES;
	
	public static boolean training = false;
	
	public Autoencoder(String path) {
//		loadNumberImages();
		loadFaceImages();
//		loadValidationImages();
		initNN(path);
//		nn.COSTFUNCTION = CostFunctions.CROSSENTROPY;
		nn.lambda = 0.001;
		if(batchsize > IMAGES.length) batchsize = IMAGES.length;
		System.out.println(WIDTH+","+HEIGHT);
	}
	
	public void initNN(String path) {
		if(path == null) {
			nn = new NeuralNetwork(WIDTH*HEIGHT,200,numOfFeatures,200,WIDTH*HEIGHT);
			nn.setActivations(new int[] {0,0,0,0});
		} else try {
			nn = new NeuralNetwork(new BufferedInputStream(getClass().getResourceAsStream(path)));
		} catch(IOException e) {
			e.printStackTrace();
		}
		window = new Window(this);
	}
	
//	boolean finished = false;
	public void loadFaceImages() {

		System.out.println("======================\nLOADING TRAINING SET\n======================");
		//Find files
		DirectoryScanner scanner = new DirectoryScanner();
		scanner.setIncludes(new String[] {"**/*_2.pgm"});
//		scanner.setExcludes(new String[] {"**/*Ambient.pgm","yaleB39/*.pgm"});
//		scanner.setExcludes(new String[] {"**/*_2.pgm", "**/*_4.pgm"});
		scanner.setBasedir("C:\\Users\\usuario\\workspace\\NEURALNET\\Neural Network\\res\\faces\\faces");
		scanner.scan();
		String[] files = scanner.getIncludedFiles();
		IMAGES = new double[files.length][];
		System.out.println(files.length);
		for(int file = 0; file < files.length; file++) {
			System.out.println("/faces/faces/"+files[file]);
			try {
				IMAGES[file] = new LoadImage().pgm("/faces/faces/"+files[file]);
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		}
//		finished = true;
	}
	
	public void loadValidationImages() {
		System.out.println("======================\nLOADING VALIDATION SET\n======================");
		DirectoryScanner scanner = new DirectoryScanner();
		scanner.setIncludes(new String[] {"yaleB39/*.pgm"});
		scanner.setExcludes(new String[] {"yaleB39/*Ambient.pgm"});
		scanner.setBasedir("C:\\Users\\usuario\\workspace\\NEURALNET\\Neural Network\\res\\faces\\CroppedYale");
		scanner.scan();
		String[] files = scanner.getIncludedFiles();
		VALIDATIONIMAGES = new double[files.length][];
		
		for(int file = 0; file < files.length; file++) {
			System.out.println("/faces/CroppedYale/"+files[file]);
			try {
				VALIDATIONIMAGES[file] = new LoadImage().pgm("/faces/CroppedYale/"+files[file]);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public void loadNumberImages() {
		BufferedInputStream imagesBuffer = new BufferedInputStream(getClass().getResourceAsStream("/train-images.idx3-ubyte"));
		IMAGES = new double[60000][28*28];
		WIDTH = 28;
		HEIGHT = 28;
		try {
			imagesBuffer.skip(16);
			
			for(int i = 0; i < IMAGES.length; i++) {
				for(int j = 0; j < IMAGES[i].length; j++) {
					IMAGES[i][j] = imagesBuffer.read()/255.0D;
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		BufferedInputStream vaBuffer = new BufferedInputStream(getClass().getResourceAsStream("/t10k-images.idx3-ubyte"));
		VALIDATIONIMAGES = new double[10000][28*28];
		try {
			vaBuffer.skip(16);
			
			for(int i = 0; i < VALIDATIONIMAGES.length; i++) {
				for(int j = 0; j < VALIDATIONIMAGES[i].length; j++) {
					VALIDATIONIMAGES[i][j] = vaBuffer.read()/255.0D;
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
//		new Autoencoder("/faces/faces/encoder.gaconvi");
		new Autoencoder(null);
	}
	
	int b = 0;
	public void trainNN() {
		double[][] pixels = new double[batchsize][WIDTH*HEIGHT];
		System.out.println(WIDTH+", " + HEIGHT);
		
		train:for(int a = 0; true; a++) {
			shuffleArray(IMAGES);
			for(int batch = b; batch < IMAGES.length/batchsize; batch++) {
				for(int i = 0; i < batchsize; i++) {
					for(int j = 0; j < pixels[i].length; j++) {
						pixels[i][j] = IMAGES[batch*batchsize+i][j];
					}
				}
				nn.batchTrain(pixels, pixels);
				if(!training) break train;
				if(batch%400==399)testNN();
			}
			b = 0;
			System.err.println(a);
		}
	}
	
	double validationcost = 0;
	
	public void testNN() {
		if(VALIDATIONIMAGES==null) return;
		double cost = 0;
		
		for(int i = 0; i < VALIDATIONIMAGES.length; i++) {
			nn.feedForward(VALIDATIONIMAGES[i]);
			cost += nn.computeCost(VALIDATIONIMAGES[i]);
		}
		System.out.println("Validation Cost: " + cost);
		validationcost = cost;
	}
	
	private void shuffleArray(double[][] array) {
		
		int index;
		double[] temp;
		Random rand = new Random();
		for(int i = 0; i < array.length; i++) {
			index = rand.nextInt(array.length);
			if(index != i) {
				temp = array[index];
				array[index] = array[i];
				array[i] = temp;
			}
		}
	}
	
	public double[][] encodeImages() {
		double[][] encodedImages = new double[IMAGES.length+(VALIDATIONIMAGES!=null ? VALIDATIONIMAGES.length:0)][numOfFeatures];
		for(int i = 0; i < IMAGES.length; i++) {
			double[] encode = nn.feed(IMAGES[i]);
			System.arraycopy(encode, 0, encodedImages[i], 0, encode.length);
		}
		if(VALIDATIONIMAGES!=null)
		for(int i = 0; i < VALIDATIONIMAGES.length; i++) {
			double[] encode = nn.feed(VALIDATIONIMAGES[i]);
			System.arraycopy(encode, 0, encodedImages[i+IMAGES.length], 0, encode.length);
		}
		System.out.println("encode");
		return encodedImages;
	}
}
