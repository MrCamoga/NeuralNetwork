package com.camoga.cnn;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import javax.imageio.ImageIO;

public class ConvNet {
	
	static ArrayList<ICLayer> cnnlayers = new ArrayList<>();
	FullyConnectedLayer fc;
	
	public ConvNet() {
		loadImages();
		cnnlayers.add(new InputLayer(1,100));
		cnnlayers.add(new ConvLayer((ICLayer)cnnlayers.get(0), 10, 5, "same", 1));
		cnnlayers.add(new ReLULayer(cnnlayers.get(1)));
		cnnlayers.add(new PoolLayer(cnnlayers.get(2)));

		cnnlayers.add(new ConvLayer((ICLayer)cnnlayers.get(3), 10, 5, "same", 1));
		cnnlayers.add(new ReLULayer(cnnlayers.get(4)));
		cnnlayers.add(new PoolLayer(cnnlayers.get(5)));

		cnnlayers.add(new ConvLayer((ICLayer)cnnlayers.get(6), 10, 5, "same", 1));
		cnnlayers.add(new ReLULayer(cnnlayers.get(7)));
		cnnlayers.add(new PoolLayer(cnnlayers.get(8)));

		cnnlayers.add(new ConvLayer((ICLayer)cnnlayers.get(9), 20, 5, "same", 1));
		cnnlayers.add(new ReLULayer(cnnlayers.get(10)));
		cnnlayers.add(new PoolLayer(cnnlayers.get(11)));
		
		ICLayer last = cnnlayers.get(cnnlayers.size()-1);
		
		fc = new  FullyConnectedLayer(
				last.depth()*last.size()*last.size(),
				15,2);
//		layers.add(new PoolLayer(layers.get(1)));
//		layers.add(new ConvLayer(layers.get(0), 5, 3, 1, 1));
//		layers.add(new PoolLayer(layers.get(1)));
		
		new Window();
		Random r = new Random();
		while(true) {
			int correct = 0;
			for(int batch = 0; batch < 3; batch++) {				
				int rand = r.nextInt(IMAGES.length);
				double[][][] input = IMAGES[rand];
				for(int i = 0; i < cnnlayers.size(); i++) {
					input = cnnlayers.get(i).forward(input);
//				System.out.println(Arrays.deepToString(input).replace("],", "],\n"));
				}
				double[] output = fc.forward(input);
				double[] target = new double[] {rand%2,(rand+1)%2};
				System.out.println(Arrays.toString(output) + ", " + Arrays.toString(target));
				double[][][] dj = fc.backprop(cnnlayers.get(cnnlayers.size()-1), target);
				
				int index = output[0] > output[1] ? 0:1;
				
				if(target[index] == 1) correct++;
				
				
				for(int i = cnnlayers.size()-1; i > 0; i--) {
					dj = cnnlayers.get(i).backprop(cnnlayers.get(i-1), dj);
				}
			}
			System.out.println("Correct: " + correct*100/3 + "%");
			gradientDescent();
			
//			System.out.println(Arrays.deepToString(djdk).replace("],", "],\n"));
			
		}
//		for(int i = layers.size()-1; i >= 0; i--) {
//			layer = layers.get(i).backprop(input);
//		}
	}
	
	public void gradientDescent() {
		for(int i = 0; i < cnnlayers.size(); i++) {
			if(!(cnnlayers.get(i) instanceof ConvLayer)) continue;
			ConvLayer layer = ((ConvLayer)cnnlayers.get(i));
			double[][][][] djdk = layer.djdk;
			for(int m = 0; m < djdk.length; m++) {
				for(int n = 0; n < djdk[0].length; n++) {
					for(int y = 0; y < djdk[0][0].length; y++) {
						for(int x = 0; x < djdk[0][0][0].length; x++) {
							layer.kernel[m][n][y][x] -= 0.0001*djdk[m][n][y][x];
						}
					}
				}
			}
			layer.djdk = new double[layer.djdk.length][layer.djdk[0].length][layer.djdk[0][0].length][layer.djdk[0][0].length];
		}
		
		fc.gradientDescent();
	}

	double[][][][] IMAGES;
	double[][] TARGETS;
	
	public void loadImages() {
		IMAGES = new double[100][3][100][100];
		for(int i = 0; i < 50; i++) {
			try {
				BufferedImage cat = ImageIO.read(getClass().getResourceAsStream("/CNN/catdog/PetImages/Cat/"+i+".png"));
				for(int y = 0; y < cat.getHeight(); y++) {
					for(int x = 0; x < cat.getWidth(); x++) {
						int rgb = cat.getRGB(x, y);
						IMAGES[2*i][0][y][x] = ((rgb&0xff0000)>>16)/255.0f;
						IMAGES[2*i][1][y][x] = ((rgb&0xff00)>>8)/255.0f;
						IMAGES[2*i][2][y][x] = (rgb&0xff)/255.0f;
					}
				}
				BufferedImage dog = ImageIO.read(getClass().getResourceAsStream("/CNN/catdog/PetImages/Dog/"+i+".png"));
				for(int y = 0; y < dog.getHeight(); y++) {
					for(int x = 0; x < dog.getWidth(); x++) {
						int rgb = dog.getRGB(x, y);
						IMAGES[2*i+1][0][y][x] = ((rgb&0xff0000)>>16)/255.0f;
						IMAGES[2*i+1][1][y][x] = ((rgb&0xff00)>>8)/255.0f;
						IMAGES[2*i+1][2][y][x] = (rgb&0xff)/255.0f;
					}
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
//	public double[][][][] IMAGES;
//	public double[][] TARGETS;
	
//	public void loadImages() {
//		BufferedInputStream imagesBuffer = new BufferedInputStream(getClass().getResourceAsStream("/train-images.idx3-ubyte"));
//		BufferedInputStream labelsBuffer = new BufferedInputStream(getClass().getResourceAsStream("/train-labels.idx1-ubyte"));
//
//		int width = 28;
//		int height = 28;
//		IMAGES = new double[6000][1][height][width];
//		TARGETS = new double[6000][10];
//		try {
//			imagesBuffer.skip(16);
//			labelsBuffer.skip(8);
//			for(int i = 0; i < 6000; i++) {
//				for(int y = 0; y < height; y++) {
//					for(int x = 0; x < width; x++) {
//						IMAGES[i][0][y][x] = imagesBuffer.read()/255.0f;
//					}
//				}
//				int label = labelsBuffer.read();
//				TARGETS[i][label] = 1;
//			}
//		} catch (IOException e) {
//			e.printStackTrace();
//		}		
//	}
	
	public static void main(String[] args) {
		new ConvNet();
	}
}