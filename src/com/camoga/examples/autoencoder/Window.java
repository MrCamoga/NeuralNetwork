package com.camoga.examples.autoencoder;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Random;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import com.camoga.nn.NeuralNetwork;

public class Window extends JFrame implements ActionListener {
	
	private Autoencoder main;
	
	public JSlider[] sliders;
	
	public boolean encoder = true;
	public boolean decoder = true;
	
	public boolean showNN = true, showImages = true, showCost = true;
	
	public Window(Autoencoder main) {
		this(800,600);
		this.main = main;
	}
	
	class Panel extends JPanel {
		public Panel() {
			
		}
		int timer = 0;
		int[] pixels, pixels2;
		BufferedImage image, image2;
		
		int imagezoom = 4;
		protected void paintComponent(Graphics g) {
			timer++;
			super.paintComponent(g);
			if(decoder && !encoder) main.nn.feed(getSliderValues());
//			BufferedImage result = new BufferedImage(2000, 18000, BufferedImage.TYPE_INT_ARGB);
//			Graphics g2 = result.getGraphics();
			if(showNN)main.nn.renderNN(g, 20, 0, 1200, 800);
			g.setColor(Color.RED);
			g.drawString("Validation cost: " + main.validationcost, 20, 20);
			
//			try {
//				ImageIO.write(result, "PNG", new File("big_autoencoderNN.png"));
//				System.err.println("SAVED");
//				System.exit(0);
//			} catch (IOException e) {
//				e.printStackTrace();
//			}
			
			
			if(showCost)main.nn.renderCostPlot(g, 500, 120, 700, 350, 0xffff0000);
			if(showImages) {
				if(encoder) {
					pixels = new int[Autoencoder.WIDTH*Autoencoder.HEIGHT];
					for(int i = 0; i < pixels.length; i++) {
						pixels[i] = (int)(main.nn.a[0][i]*255)*0x10101;
					}
					image = new BufferedImage(Autoencoder.WIDTH, Autoencoder.HEIGHT, BufferedImage.TYPE_INT_RGB);
					image.setRGB(0, 0, Autoencoder.WIDTH, Autoencoder.HEIGHT, pixels, 0, Autoencoder.WIDTH);				
					g.drawImage(image, 500, 500, Autoencoder.WIDTH*imagezoom, Autoencoder.HEIGHT*imagezoom, null);
				}
				if(decoder) {
					pixels2 = new int[Autoencoder.WIDTH*Autoencoder.HEIGHT];
					for(int i = 0; i < pixels2.length; i++) {
						pixels2[i] = (int)(main.nn.a[main.nn.a.length-1][i]*255)*0x10101;
					}
					image2 = new BufferedImage(Autoencoder.WIDTH, Autoencoder.HEIGHT, BufferedImage.TYPE_INT_RGB);
					image2.setRGB(0, 0, Autoencoder.WIDTH, Autoencoder.HEIGHT, pixels2, 0, Autoencoder.WIDTH);				
					g.drawImage(image2, 500+Autoencoder.WIDTH*imagezoom, 500, Autoencoder.WIDTH*imagezoom, Autoencoder.HEIGHT*imagezoom, null);
				}				
			}
			
//			if(main.finished) {
//				BufferedImage image3 = new BufferedImage(Autoencoder.WIDTH, Autoencoder.HEIGHT, BufferedImage.TYPE_INT_RGB);
//				int[] pixels3 = new int[main.IMAGES[timer%main.IMAGES.length].length];
//				for(int i = 0; i < pixels3.length; i++) {
//					pixels3[i] = (int)(main.IMAGES[timer%main.IMAGES.length][i]*255)*0x10101;
//				}
//				image3.setRGB(0, 0, Autoencoder.WIDTH, Autoencoder.HEIGHT, pixels3, 0, Autoencoder.WIDTH);
//				g.drawImage(image3, 0, 0, Autoencoder.WIDTH*4, Autoencoder.HEIGHT*4, null);				
//			}
			if(encodedImages!=null) {
				g.setColor(Color.black);
				g.drawLine(500, 200, 500, 300);
				g.drawLine(500, 200, 600, 200);
				g.drawLine(600, 200, 600, 300);
				g.drawLine(500, 300, 600, 300);
				g.setColor(Color.red);
//				System.out.println(timer);
				for(int i = 0; i < encodedImages[(timer/10)%80].length; i++) {
					for(int j = 0; j < encodedImages[(timer/10+1)%80].length; j++) {
						g.fillOval(500+(int)(encodedImages[(timer/10)%80][i]*200), (int)(200+encodedImages[(timer/10+1)%80][j]*200), 3, 3);
					}
				}
			}
			
			repaint();
		}
		
	}
	
	public double[] getSliderValues() {
		double[] inputs = new double[main.numOfFeatures];
		for(int i = 0; i < sliders.length; i++) {
			inputs[i] = sliders[i].getValue()/100.0D;
		}
		return inputs;
	}
		
	public Window(int width, int height) {
		super("Number Autoencoder - by MrCamoga");
		
		setSize(width, height);
		setResizable(true);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		JMenuBar menuBar = new JMenuBar();
			JMenu file = new JMenu("File");
				JMenuItem savecost = new JMenuItem("Save Cost");
				savecost.addActionListener(this);
			file.add(savecost);
		
		menuBar.add(file);
		
			JMenu net = new JMenu("Neural Network");
				JMenuItem newnn = new JMenuItem("New NN");
				JMenuItem open = new JMenuItem("Open");
				JMenuItem save = new JMenuItem("Save");
				
				newnn.addActionListener(this);
				open.addActionListener(this);
				save.addActionListener(this);
				
			net.add(newnn);
			net.add(open);
			net.add(save);
				
				JMenu training = new JMenu("Training");
					JMenuItem learning = new JMenuItem("Update lambda");
					JMenuItem train = new JMenuItem("Start training");
					learning.addActionListener(this);
					train.addActionListener(this);
				training.add(learning);
				training.add(train);
				
			net.add(training);
		menuBar.add(net);
			
			JMenu autoencoder = new JMenu("Autoencoder");
				JMenuItem sencoder = new JMenuItem("Save Encoder");
				JMenuItem sdecoder = new JMenuItem("Save Decoder");
				JMenuItem lencoder = new JMenuItem("Load Encoder");
				JMenuItem ldecoder = new JMenuItem("Load Decoder");
				JMenuItem encodeImages = new JMenuItem("Encode Images");
				sencoder.addActionListener(this);
				sdecoder.addActionListener(this);
				lencoder.addActionListener(this);
				ldecoder.addActionListener(this);
				encodeImages.addActionListener(this);
			autoencoder.add(sencoder);
			autoencoder.add(sdecoder);
			autoencoder.add(lencoder);
			autoencoder.add(ldecoder);
			autoencoder.add(encodeImages);
		menuBar.add(autoencoder);
			
			JMenu appearance = new JMenu("Appearance");
				JMenuItem switchImages = new JMenuItem("Switch Images");
				JMenuItem switchCost = new JMenuItem("Switch Cost");
				JMenuItem switchNN = new JMenuItem("Switch NN");
				switchCost.addActionListener(this);
				switchImages.addActionListener(this);
				switchNN.addActionListener(this);
			appearance.add(switchCost);
			appearance.add(switchImages);
			appearance.add(switchNN);
		menuBar.add(appearance);
			
		add(menuBar, BorderLayout.NORTH);
		
		Panel panel = new Panel();
		add(panel);
		
		sliders = new JSlider[main.numOfFeatures];
		for(int i = 0; i < sliders.length; i++) {
			sliders[i] = new JSlider(SwingConstants.VERTICAL, 0, 100, 50);
		}
		JButton randize = new JButton("Randomize");
		Random r = new Random(20);
		randize.addActionListener(new ActionListener() {
			
			public void actionPerformed(ActionEvent e) {
				for(JSlider s : sliders) {
					s.setValue((int) (r.nextGaussian()*50+50));
				}
				
			}
		});
		JPanel sliderPanel = new JPanel(new GridLayout(4,20));
		for(JSlider s : sliders) {
			sliderPanel.add(s);
		}
		panel.add(randize);
		add(sliderPanel,BorderLayout.EAST);

		setVisible(true);
	}
	
	Thread train;

	public void actionPerformed(ActionEvent e) {
		switch (e.getActionCommand()) {
			case "New NN":
				JPanel panel = new JPanel(new GridLayout(6, 2));
				JTextField i = new JTextField();
				panel.add(new JLabel("i:"));
				panel.add(i);
				JTextField h = new JTextField();
				panel.add(new JLabel("h:"));
				panel.add(h);
				JTextField h1 = new JTextField();
				panel.add(new JLabel("h2:"));
				panel.add(h1);
				JTextField o = new JTextField();
				panel.add(new JLabel("o:"));
				panel.add(o);
				JTextField batch = new JTextField();
				panel.add(new JLabel("batch:"));
				panel.add(batch);
				JTextField lambda = new JTextField();
				panel.add(new JLabel("lambda: "));
				panel.add(lambda);
				int result = JOptionPane.showConfirmDialog(this, panel, "Create new NN", JOptionPane.OK_CANCEL_OPTION);
				if(result == JOptionPane.OK_OPTION) {
					//TODO create NN
					main.nn = new NeuralNetwork(Integer.parseInt(i.getText()), Integer.parseInt(h.getText()), Integer.parseInt(h1.getText()), Integer.parseInt(o.getText()));
				}
				break;
			case "Open":
				openState();
				encoder = true;
				decoder = true;
				break;
			case "Save":
				saveNN();
				break;
			case "Update lambda":
				JPanel panel2 = new JPanel(new GridLayout(1, 2));
				panel2.add(new JLabel("Learning rate (lambda"));
				JTextField l = new JTextField(main.nn.lambda+"");
				panel2.add(l);
				int result2 = JOptionPane.showConfirmDialog(this, panel2, "Update Learning Rate", JOptionPane.OK_CANCEL_OPTION);
				if(result2 == JOptionPane.OK_OPTION) main.nn.lambda = Double.parseDouble(l.getText());
				break;
			case "Start training":
				main.training = true;
				((JMenuItem)e.getSource()).setText("Stop training");
				train = new Thread(()->main.trainNN());
				train.start();
				break;
			case "Stop training":
				main.training = false;
				((JMenuItem)e.getSource()).setText("Start training");
				try {
					train.join();
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}
				break;
			case "Save Encoder":
				saveEncoder();
				
				break;
			case "Save Decoder":
				saveDecoder();
				break;
			case "Load Encoder":
				openState();
				encoder = true;
				decoder = false;
				break;
			case "Load Decoder":
				openState();
				encoder = false;
				decoder = true;
				break;
			case "Encode Images":
				if(encoder&&!decoder) encode();
				break;
			case "Save Cost":
				String path = getSavePath();
				if(path != null)
				main.nn.saveCost(path);
				break;
			case "Switch Cost":
				showCost = !showCost;
				break;
			case "Switch NN":
				showNN = !showNN;
				break;
			case "Switch Images":
				showImages = !showImages;
				break;
			}
	}
	
	public void openState() {
		JFileChooser file = new JFileChooser("C:/Users/Yolanda1/Desktop/workspace/Neural Network/");
		int option = file.showOpenDialog(this);
		if(option == JFileChooser.CANCEL_OPTION) return;
		try {
			String path = file.getSelectedFile().getPath();
			System.out.println(path);
			main.nn = new NeuralNetwork(new BufferedInputStream(new FileInputStream(file.getSelectedFile())));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	double[][] encodedImages;
	double[][] covariance;
	public void encode() {
		if(encodedImages == null) encodedImages = main.encodeImages();
		int[][] distribution;
		distribution = new int[80][20];
		for(int j = 0; j < encodedImages[0].length; j++) {
			for(int i = 0; i < encodedImages.length; i++) {
				double value = encodedImages[i][j];
				distribution[j][(int)((value+1)*10)]++;				
			}
		}
//		for(int k = 0; k < distribution.length; k++) {
		for(int l = 0; l < distribution[6].length; l++) {
			System.out.print(distribution[6][l]+"\t");				
		}
//			System.out.println();
//		}
			
//		for(int n = 0; n < encodedImages[0].length; n++) {
//			double mean = 0;
//			for(int i = 0; i < encodedImages.length; i++) {
//				mean += encodedImages[i][n];				
//			}
//			mean /= (double)encodedImages.length;
//			double sum = 0;
//			for(int i = 0; i < encodedImages.length; i++) {
//				sum += Math.pow(encodedImages[i][n]-mean,2);
//			}
//			sum /= (double)encodedImages.length;
//			
//			double sd = Math.sqrt(sum);
//			System.out.println("Distribution "+ n+": sd: " + sd+ ", var: " + sum + ", mean: " + mean);
//		}
		covariance = new double[main.numOfFeatures][main.numOfFeatures];
		double[] mean = new double[main.numOfFeatures];
		double m = 0;
		for(int x = 0; x < main.numOfFeatures; x++) {
			for(int i = 0; i < encodedImages.length; i++) {
				m += encodedImages[i][x];
			}
			m /= (double)encodedImages.length;
			mean[x] = m;
		}
		
		for(int j = 0; j < covariance.length; j++) {
			for(int k = 0; k < covariance[j].length; k++) {
				double sum = 0;
				for(int i = 0; i < encodedImages.length; i++) {
					sum = (encodedImages[i][j]-mean[j])*(encodedImages[i][k]-mean[k]);
				}
				sum /= encodedImages.length-1;
				covariance[j][k] = sum;
			}
		}
		
//		for(int j = 0; j < covariance.length; j++) {
//			System.out.print("{{");
//			for(int k = 0; k < covariance[j].length; k++) {
//				System.out.print(covariance[j][k] + ", ");
//			}
//			System.out.println("},");
//		}
	}
	
	public void saveEncoder() {
		String path = getSavePath();
		if(path != null)
		main.nn.saveLayers(0, main.nn.a.length/2+1, path);
	}
	
	public void saveDecoder() {
		String path = getSavePath();
		if(path != null)
		main.nn.saveLayers(main.nn.a.length/2, main.nn.a.length, path);
	}
	
	public void saveNN() {
		String path = getSavePath();
		if(path != null)
		main.nn.save(path);
	}
	
	public String getSavePath() {
		JFileChooser file = new JFileChooser();
		int state = file.showSaveDialog(this);
		if(state == JFileChooser.CANCEL_OPTION) return null;
		return file.getSelectedFile().getAbsolutePath();
	}
}
