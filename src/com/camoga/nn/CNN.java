package com.camoga.nn;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.apache.tools.ant.taskdefs.Sleep;

public class CNN {
	
	private int width = 1200, height = 750;
	
	//TODO more than one image on input (input[][][] and filter[][][][])
	double[][] input;
	double[][][][] kernel1, kernel2, kernel3;
	double[][][] conv1, relu1, sample1, conv2, relu2, conv3, relu3, sample2;
	NeuralNetwork nn;
	
	private double lambda = 0.00001;
	
	class Panel extends JPanel {
		
		int zoom = 1;
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);
			
			BufferedImage img = createImage(input);
			g.drawImage(img, 0, 50*zoom, img.getWidth()*zoom, img.getHeight()*zoom, null);
			
			nn.renderCostPlot(g, 80, 500, 300, 200, 0xffff0000);
			
//			BufferedImage img2 = createImage(convolution);
//			g.drawImage(img2, img.getWidth()*16, (img.getHeight()-img2.getHeight())*8,img2.getWidth()*16, img2.getHeight()*16, null);
		
		
//			BufferedImage img3 = createImage(subsample);
//			g.drawImage(img3, 0, img.getHeight()*16,img3.getWidth()*16, img3.getHeight()*16, null);
			for(int i = 0; i < kernel1.length; i++) {
				for(int j = 0; j < kernel1[i].length; j++) {
					BufferedImage img2 = createImage(kernel1[i][j]);
					g.drawImage(img2, 120+(1+img2.getWidth())*zoom*i*4, (6+img2.getHeight())*zoom*j*4,img2.getWidth()*zoom*4, img2.getHeight()*zoom*4, null);					
				}
			}

			for(int i = 0; i < conv1.length; i++) {
				BufferedImage img2 = createImage(relu1[i]);
				g.drawImage(img2, 200, (20+img2.getHeight())*zoom*i,img2.getWidth()*zoom, img2.getHeight()*zoom, null);
			}
			
			
			for(int i = 0; i < sample1.length; i++) {
				BufferedImage img2 = createImage(sample1[i]);
				g.drawImage(img2, 300, (20+img2.getHeight())*zoom*i,img2.getWidth()*zoom*2, img2.getHeight()*zoom*2, null);
			}
			
			for(int i = 0; i < kernel2.length; i++) {
				for(int j = 0; j < kernel2[i].length; j++) {
					BufferedImage img2 = createImage(kernel2[i][j]);
					g.drawImage(img2, 400+(1+img2.getWidth())*zoom*i*4, (1+img2.getHeight())*zoom*j*4,img2.getWidth()*zoom*4, img2.getHeight()*zoom*4, null);
				}
			}
			
			for(int i = 0; i < conv2.length; i++) {
				BufferedImage img2 = createImage(relu2[i]);
				g.drawImage(img2, 800, (20+img2.getHeight())*zoom*i,img2.getWidth()*zoom*2, img2.getHeight()*zoom*2, null);
			}
			
			for(int i = 0; i < kernel3.length; i++) {
				for(int j = 0; j < kernel3[i].length; j++) {
					BufferedImage img2 = createImage(kernel3[i][j]);
					g.drawImage(img2, 400+(1+img2.getWidth())*zoom*i*4, 400+(1+img2.getHeight())*zoom*j*4,img2.getWidth()*zoom*4, img2.getHeight()*zoom*4, null);
				}
			}
			
			for(int i = 0; i < conv3.length; i++) {
				BufferedImage img2 = createImage(relu3[i]);
				g.drawImage(img2, 730, 400+(20+img2.getHeight())*zoom*i,img2.getWidth()*zoom*2, img2.getHeight()*zoom*2, null);
			}
			
			for(int i = 0; i < sample2.length; i++) {
				BufferedImage img2 = createImage(sample2[i]);
				g.drawImage(img2, 880, (20+img2.getHeight())*zoom*i,img2.getWidth()*zoom*4, img2.getHeight()*zoom*4, null);
			}
			
			nn.renderNN(g, 980, 0, 500, 360);
			repaint();
		}
		
		public BufferedImage createImage(double[][] image) {
			int[] pixels = new int[image.length*image[0].length];
			BufferedImage img = new BufferedImage(image.length, image[0].length, BufferedImage.TYPE_INT_RGB);
			for(int y = 0; y < image.length; y++) {
				for(int x = 0; x < image[0].length; x++) {
					pixels[x+y*image.length] = (int)(Math.abs(image[y][x])*0xff)*0x10101;
				}
			}
			img.setRGB(0, 0, image[0].length, image.length, pixels, 0, image[0].length);
			return img;
		}
	}
	
	class Window {
		public Window() {
			JFrame f = new JFrame("Convolutional Neural Network by MrCamoga");
			f.setSize(width,height);
			f.setResizable(true);
			f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			f.setLocationRelativeTo(null);
			
			
			Panel panel = new Panel();
			f.add(panel);
			
			f.setVisible(true);
		}
	}
	
	public CNN() {
		loadNumberImages();
		
		int inputimages = 1;
		//FIXME outputimage1 cant be greater than outputimage2
		int outputimages1 = 6;
		int outputimages2 = 18;
		int outputimages3 = 14;
		int kernelSize = 5;
		int imagewidth = 28, imageheight = 28;
		
		kernel1 = new double[inputimages][outputimages1][kernelSize][kernelSize];
		conv1 = new double[outputimages1][imageheight][imagewidth];
		relu1 = new double[conv1.length][conv1[0].length][conv1[0][0].length];
		sample1 = new double[conv1.length][(int)Math.ceil(conv1[0].length/2)][(int)Math.ceil(conv1[0][0].length/2)];
		
		kernel2 = new double[sample1.length][outputimages2][kernelSize][kernelSize];
		conv2 = new double[outputimages2][sample1[0].length][sample1[0][0].length];
		relu2 = new double[conv2.length][conv2[0].length][conv2[0][0].length];
		
		kernel3 = new double[conv2.length][outputimages3][kernelSize][kernelSize];
		conv3 = new double[outputimages3][conv2[0].length][conv2[0][0].length];
		relu3 = new double[conv3.length][conv3[0].length][conv3[0][0].length];
		
		sample2 = new double[conv3.length][(int)Math.ceil(conv3[0].length/2)][(int)Math.ceil(conv3[0][0].length/2)];
		
		nn = new NeuralNetwork(sample2.length*sample2[0].length*sample2[0][0].length,20,20,10);
		nn.setActivations(new int[3]);
		nn.lambda = 0.0001;
		
		//Random init kernels
		Random r = new Random();
		for(int m = 0; m < kernel1.length; m++) {
			for(int n = 0; n < kernel1[m].length; n++) {
				for(int i = 0; i < kernel1[m][n].length; i++) {
					for(int j = 0; j < kernel1[m][n][i].length; j++) {
						kernel1[m][n][i][j] = r.nextGaussian();
//					if(filter1[n][i][j] > 1) filter1[n][i][j] = 1;
					}
				}
				
			}
		}
		for(int n = 0; n < kernel2.length; n++) {
			for(int m = 0; m < kernel2[n].length; m++) {
				for(int i = 0; i < kernel2[n][m].length; i++) {
					for(int j = 0; j < kernel2[n][m][i].length; j++) {
						kernel2[n][m][i][j] = r.nextGaussian();
					}
				}
				
			}
		}
		
		for(int n = 0; n < kernel3.length; n++) {
			for(int m = 0; m < kernel3[n].length; m++) {
				for(int i = 0; i < kernel3[n][m].length; i++) {
					for(int j = 0; j < kernel3[n][m][i].length; j++) {
						kernel3[n][m][i][j] = r.nextGaussian();
					}
				}
				
			}
		}
		
		new Window();
		int batchSize = 30;
		for(int a = 0; true; a++) {
			for(int batch = 0; batch < IMAGES.length/batchSize; batch++) {
				double[][][] batchImages = new double[batchSize][][];
				double[][] batchTargets = new double[batchSize][];
				for(int i = 0; i < batchSize; i++) {
//					System.out.println("{");
//					for(int m = 0; m < kernel1[0][0].length; m++) {
//						System.out.print("{");
//						for(int n = 0; n < kernel1[0][0][m].length; n++) {
//							System.out.print(kernel1[0][0][m][n]+", ");
//						}
//						System.out.println("},");
//					}
//					System.out.println("}");
//					train(IMAGES[i], TARGETS[i]);
					batchImages[i] = IMAGES[batch*batchSize+i];
					batchTargets[i] = TARGETS[batch*batchSize+i];
				}	
				batchTrain(batchImages, batchTargets);
			}
		}
	}
	
	//TODO backprop in single training 
	public void train(double[][] image, double[] target) {
		this.input = image;
		for(int m = 0; m < kernel1.length; m++) {
			for(int n = 0; n < kernel1[m].length; n++) {
				conv1[m] = convolve(input, kernel1[m][n]);				
			}
		}
		
		//Relu
		for(int n = 0; n < conv1.length; n++) {
			relu(conv1[n]);
		}
		
		
		for(int n = 0; n < sample1.length; n++) {
			sample1[n] = subsampling(conv1[n]);
		}
		
		for(int n = 0; n < kernel2.length; n++) {
			for(int m = 0; m < kernel2[n].length; m++) {
				if(n == 0) conv2[m] = convolve(sample1[n], kernel2[n][m]);
				else conv2[m] = add(conv2[m], convolve(sample1[n], kernel2[n][m]));
			}
		}
		
		for(int n = 0; n < conv2.length; n++) {
			relu(conv2[n]);
		}
		
		for(int n = 0; n < kernel3.length; n++) {
			for(int m = 0; m < kernel3[n].length; m++) {
				if(n == 0) conv3[m] = convolve(conv2[n], kernel3[n][m]);
				else conv3[m] = add(conv3[m], convolve(conv2[n], kernel3[n][m]));
			}
		}
		
		for(int n = 0; n < conv3.length; n++) {
			relu(conv3[n]);
		}
		
		for(int n = 0; n < sample2.length; n++) {
			sample2[n] = subsampling(conv3[n]);
		}
		
		double[] nninput = new double[sample2.length*sample2[0].length*sample2[0][0].length];
		int index = 0;
		for(int n = 0; n < sample2.length; n++) {
			for(int i = 0; i < sample2[n].length; i++) {
				for(int j = 0; j < sample2[n][i].length; j++) {
					nninput[index] = sample2[n][i][j];
					index++;
				}
			}
		}
		
		double[] dJda = nn.train(nninput, target);
		
	}
	
	public void batchTrain(double[][][] image, double[][] target) {
		double[][] nninput = new double[image.length][sample2.length*sample2[0].length*sample2[0][0].length];
		
		double[][][][] dJdK3 = new double[kernel3.length][kernel3[0].length][kernel3[0][0].length][kernel3[0][0][0].length];
		double[][][][] dJdK2 = new double[kernel2.length][kernel2[0].length][kernel2[0][0].length][kernel2[0][0][0].length];
		double[][][][] dJdK1 = new double[kernel1.length][kernel1[0].length][kernel1[0][0].length][kernel1[0][0][0].length];
		
		for(int i = 0; i < image.length; i++) {
			this.input = image[i];
			for(int m = 0; m < kernel1.length; m++) {
				for(int n = 0; n < kernel1[m].length; n++) {
					conv1[n] = convolve(input, kernel1[m][n]);					
				}
			}
			
			//Relu
			for(int n = 0; n < conv1.length; n++) {
				relu1[n] = relu(conv1[n]);
			}
			
			for(int n = 0; n < sample1.length; n++) {
				sample1[n] = subsampling(relu1[n]);
			}
			
			//FIXME convolution may have blue pixels
			for(int m = 0; m < kernel2.length; m++) {
				for(int n = 0; n < kernel2[m].length; n++) {
					if(m == 0) conv2[n] = convolve(sample1[m], kernel2[m][n]);
					else conv2[n] = add(conv2[n], convolve(sample1[m], kernel2[m][n]));
				}
			}
			
			for(int n = 0; n < conv2.length; n++) {
				relu2[n] = relu(conv2[n]);
			}
			
			for(int m = 0; m < kernel3.length; m++) {
				for(int n = 0; n < kernel3[m].length; n++) {
					if(m == 0) conv3[n] = convolve(relu2[m], kernel3[m][n]);
					else conv3[n] = add(conv3[n], convolve(relu2[m], kernel3[m][n]));
				}
			}
			
			for(int n = 0; n < conv3.length; n++) {
				relu3[n] = relu(conv3[n]);
			}
			
			for(int n = 0; n < sample2.length; n++) {
				sample2[n] = subsampling(relu3[n]);
			}
			int index = 0;
			for(int n = 0; n < sample2.length; n++) {
				for(int y = 0; y < sample2[n].length; y++) {
					for(int x = 0; x < sample2[n][y].length; x++) {
						nninput[i][index] = sample2[n][y][x];
						index++;
					}
				}
			}
		
		//TODO IMPLEMENT CONV BIAS
		//Backpropagation
		//FIXME it's doing backprop with all the inputs
		double[][] dJda = nn.batchTrain(nninput, target);
		
//		long last = System.nanoTime();
			double[][][] dJds2 = new double[sample2.length][sample2[0].length][sample2[0][0].length];
			double[][][] dJdr3 = new double[sample2.length][][];
			int index2 = 0;
			//Third pool layer
			for(int m = 0; m < dJds2.length; m++) {
				for(int n = 0; n < dJds2[m].length; n++) {
					for(int o = 0; o < dJds2[m][n].length; o++) {
						dJds2[m][n][o] = dJda[i][index2];
						index2++;
					}
				}
				dJdr3[m] = upsampling(dJds2[m], relu3[m]);
			}
			
			//Relu3
			double[][][] dJdO3 = new double[sample2.length][][];
			for(int m = 0; m < dJdO3.length; m++) {
				dJdO3[m] = reluPrime(dJdr3[m]);
			}
			
			//Third conv layer weights
			for(int m = 0; m < kernel3.length; m++) {
				for(int n = 0; n < kernel3[m].length; n++) {
					dJdK3[m][n] = add(dJdK3[m][n],dJdK(relu2[m], dJdO3[n], kernel3[m][n]));
				}
			}
			
			double[][][] dJdr2 = new double[conv2.length][conv2[0].length][conv2[0][0].length];
			//Third conv layer input
			for(int a = 0; a < kernel3.length; a++) {
				for(int b = 0; b < kernel3[a].length; b++) {
					dJdr2[a] = add(dJdr2[a], dJdI(dJdO3[b], kernel3[a][b]));					
				}
			}
			
			//Relu2
			double[][][] dJdO2 = new double[conv2.length][conv2[0].length][conv2[0][0].length];
			for(int m = 0; m < dJdO2.length; m++) {
				dJdO2[m] = reluPrime(dJdr2[m]);
			}
			
			//Second conv layer weights
			for(int m = 0; m < kernel2.length; m++) {
				for(int n = 0; n < kernel2[m].length; n++) {
					dJdK2[m][n] = add(dJdK2[m][n],dJdK(sample1[m], dJdO2[n], kernel2[m][n]));
				}
			}
			
			double[][][] dJds1 = new double[sample1.length][sample1[0].length][sample1[0][0].length];
			double[][][] dJdr1 = new double[sample1.length][][];
			
			for(int a = 0; a < kernel2.length; a++) {
				for(int b = 0; b < kernel2[a].length; b++) {
					dJds1[a] = add(dJds1[a], dJdI(dJdO2[b], kernel2[a][b]));					
				}
			}
			
			//First pool layer
			for(int m =0; m < dJdr1.length; m++) {
				dJdr1[m] = upsampling(dJds1[m], relu1[m]);				
			}
			
			double[][][] dJdO1 = new double[conv1.length][conv1[0].length][conv1[0][0].length];
			for(int m = 0; m < dJdO1.length; m++) {
				dJdO1[m] = reluPrime(dJdr1[m]);
			}
			
			//First Conv layer
			for(int m = 0; m < kernel1.length; m++) {
				for(int n = 0; n < kernel1[m].length; n++) {
					dJdK1[m][n] = add(dJdK1[m][n],dJdK(input, dJdO1[n], kernel1[m][n]));					
				}
			}
		}
		gradientDescent(dJdK1, dJdK2, dJdK3);
		
	}
	
	public void gradientDescent(double[][][][] dJdK1, double[][][][] dJdK2, double[][][][] dJdK3) {
		for(int i = 0; i < dJdK1.length; i++) {
			for(int j = 0; j < dJdK1[i].length; j++) {
				for(int k = 0; k < dJdK1[i][j].length; k++) {
					for(int l = 0; l < dJdK1[i][j][k].length; l++) {
						kernel1[i][j][k][l] += -lambda*dJdK1[i][j][k][l];
					}
				}
			}
		}
		
		for(int i = 0; i < dJdK2.length; i++) {
			for(int j = 0; j < dJdK2[i].length; j++) {
				for(int k = 0; k < dJdK2[i][j].length; k++) {
					for(int l = 0; l < dJdK2[i][j][k].length; l++) {
						kernel2[i][j][k][l] += -lambda*dJdK2[i][j][k][l];
					}
				}
			}
		}
		
		for(int i = 0; i < dJdK3.length; i++) {
			for(int j = 0; j < dJdK3[i].length; j++) {
				for(int k = 0; k < dJdK3[i][j].length; k++) {
					for(int l = 0; l < dJdK3[i][j][k].length; l++) {
						kernel3[i][j][k][l] += -lambda*dJdK3[i][j][k][l];
					}
				}
			}
		}
	}
	
	public double[][][] IMAGES;
	public double[][] TARGETS;
	
	public void loadNumberImages() {
		BufferedInputStream imagesBuffer = new BufferedInputStream(getClass().getResourceAsStream("/train-images.idx3-ubyte"));
		BufferedInputStream labelsBuffer = new BufferedInputStream(getClass().getResourceAsStream("/train-labels.idx1-ubyte"));

		int width = 28;
		int height = 28;
		IMAGES = new double[60000][height][width];
		TARGETS = new double[60000][10];
		try {
			imagesBuffer.skip(16);
			labelsBuffer.skip(8);
			for(int i = 0; i < 60000; i++) {
				for(int y = 0; y < height; y++) {
					for(int x = 0; x < width; x++) {
						IMAGES[i][y][x] = imagesBuffer.read()/255.0D;
					}
				}
				int label = labelsBuffer.read();
				TARGETS[i][label] = 1;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		input = IMAGES[0];
		
	}
	
	//Clip edges
	public double[][] convolve2(double[][] image, double[][] kernel) {
		//This works assuming the kernel is square. If not, change the (width-kernel.length+1) by kernel[0].length
		double[][] convolution = new double[(image.length-kernel.length+1)][(image[0].length-kernel.length+1)];
		for(int y = 0; y < convolution.length; y++) {
			for(int x = 0; x < convolution[0].length; x++) {
				double result = 0;
				for(int i = 0; i < kernel.length; i++) {
					for(int j = 0; j < kernel[i].length; j++) {
						result += kernel[i][j]*image[y+i][x+j];
					}
				}
				convolution[y][x] = result;
			}
		}
		return convolution;
	}
	
	public double[][] convolve(double[][] image, double[][] kernel) {
		int kernelSize = (kernel.length-1)/2;
		//This works assuming the kernel is square. If not, change the (width-kernel.length+1) by kernel[0].length
		double[][] convolution = new double[image.length][image[0].length];
		for(int y = 0; y < convolution.length; y++) {
			for(int x = 0; x < convolution[0].length; x++) {
				double result = 0;
				for(int i = 0; i < kernel.length; i++) {
					int yo = y+i-kernelSize;
					if(yo < 0 || yo >= image.length) continue;
					for(int j = 0; j < kernel[i].length; j++) {
						int xo = x+j-kernelSize;
						if(xo < 0 || xo >= image[0].length) continue;
						result += kernel[i][j]*image[yo][xo];
					}
				}
				convolution[y][x] = result;
			}
		}
		return convolution;
	}
	
	public double[][] dJdK(double[][] input, double[][] dJdO, double[][] kernel) {
		double[][] dJdK = new double[kernel.length][kernel[0].length];
		int kernelSize = (kernel.length-1)/2;
		for(int ky = 0; ky < kernel.length; ky++) {
			for(int kx = 0; kx < kernel[ky].length; kx++) {
				for(int oy = 0; oy < dJdO.length; oy++) {
					int iy = oy + ky - kernelSize;
					if(iy < 0 || iy >= input.length) continue;
					for(int ox = 0; ox < dJdO[oy].length; ox++) {
						int ix = ox + kx - kernelSize;
						if(ix < 0 || ix >= input[iy].length) continue;
						dJdK[ky][kx] += dJdO[oy][ox]*input[iy][ix];
					}
				}
			}
		}
		return dJdK;
	}
	
	public double[][] dJdI(double[][] dJdO, double[][] kernel) {
		double[][] dJdI = new double[dJdO.length][dJdO[0].length];
		int kernelSize = (kernel.length-1)/2;
		for(int y = 0; y < dJdO.length; y++) {
			for(int x = 0; x < dJdO[y].length; x++) {
				for(int ky = 0; ky < kernel.length; ky++) {
					int yo = y + ky - kernelSize;
					if(yo < 0 || yo >= dJdO.length) continue;
					for(int kx = 0; kx < kernel[ky].length; kx++) {
						int xo = x + kx - kernelSize;	
						if(xo < 0 || xo >= dJdO[y].length) continue;
						dJdI[y][x] += dJdO[yo][xo]*kernel[kernel.length-ky-1][kernel[0].length-kx-1];
					}
				}
			}
		}
		return dJdI;
	}
	
	public double[][] add(double[][] img1, double[][] img2) {
		double[][] result = new double[img1.length][img1[0].length];
		for(int y = 0; y < img1.length; y++) {
			for(int x = 0; x < img1[y].length; x++) {
				result[y][x] = img1[y][x] + img2[y][x];
			}
		}
		return result;
	}
	
	public double[][] mul(double[][] a, double[][] b) {
		double[][] result = new double[a.length][a[0].length];
		for(int y = 0; y < a.length; y++) {
			for(int x = 0; x < a[y].length; x++) {
				result[y][x] = a[y][x] * b[y][x];
			}
		}
		return result;
	}
	
	public void sigmoid(double[][] image) {
		for(int i = 0; i < image.length; i++) {
			for(int j = 0; j < image.length; j++) {
				image[i][j] = 1/(1+Math.exp(-image[i][j]));
			}
		}
	}
	
	public double[][] sigmoidPrime(double[][] image) {
		double[][] dadz = new double[image.length][image[0].length];
		for(int i = 0; i < image.length; i++) {
			for(int j = 0; j < image.length; j++) {
				double s = 1/(1+Math.exp(-image[i][j]));
				dadz[i][j] = s*(1-s);
			}
		}
		return dadz;
	}
	
	public double[][] relu(double[][] image) {
		double[][] relu = new double[image.length][image[0].length];
		for(int i = 0; i < image.length; i++) {
			for(int j = 0; j < image.length; j++) {
				if(image[i][j] < 0) relu[i][j] = 0;	
				else if(image[i][j] > 1) relu[i][j] = 1;
				else relu[i][j] = image[i][j];
			}
		}
		return relu;
	}
	
	public double[][] reluPrime(double[][] dJ) {
		double[][] relup = new double[dJ.length][dJ[0].length];
		for(int i = 0; i < dJ.length; i++) {
			for(int j = 0; j < dJ[i].length; j++) {
				if(dJ[i][j] < 0) relup[i][j] = 0;	
				else if(dJ[i][j] < 1) relup[i][j] = 1;
				else relup[i][j] = 0;
			}
		}
		return relup;
	}
	
	//XXX may be the same!
	public double[][] subsampling(double[][] image) {
		double[][] subsample = new double[(int) Math.ceil(image.length/2)][(int) Math.ceil(image[0].length/2)];
		
		double maxvalue = 0;
		for(int y = 0; y < subsample.length; y++) {
			for(int x = 0; x < subsample[0].length; x++) {
				maxvalue = 0;
				for(int yi = 0; yi < 2; yi++) {
					for(int xi = 0; xi < 2; xi++) {
						double value = image[y*2+yi][x*2+xi];
						if(maxvalue < value) maxvalue = value;
					}
				}
				subsample[y][x] = maxvalue;
			}
		}
		
		return subsample;
	}
	
	public double[][] upsampling(double[][] dJds, double[][] original) {
		double[][] upsample = new double[original.length][original[0].length];
		int maxX, maxY;
		double maxvalue = 0;
		for(int y = 0; y < dJds.length; y++) {
			for(int x = 0; x < dJds[y].length; x++) {
				maxX = 0;
				maxY = 0;
				maxvalue = 0;
				for(int yi = 0; yi < 2; yi++) {
					int yo = y*2+yi;
					for(int xi = 0; xi < 2; xi++) {
						int xo = x*2+xi;
						double value = original[yo][xo];
						if(maxvalue < value) {
							maxvalue = value;
							maxX = xo;
							maxY = yo;
						}
					}
				}
				upsample[maxY][maxX] = dJds[y][x];
			}
		}
		
		return upsample;
	}
	
	public static void main(String[] args) {
		new CNN();
	}
}