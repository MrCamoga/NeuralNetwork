package com.camoga.nn;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.Arrays;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class Window extends JFrame implements ActionListener {
	private static final long serialVersionUID = 1L;
	
	private int width, height;
	private Paint paint;
	
	public Window() {
		this(800,600);
	}
	
	class Panel extends JPanel {
		private static final long serialVersionUID = 1L;

		public Panel() {
			
		}
		
		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);
			if(Main.nn == null) return;
//			int diameter = 20;
//			int yspacing = 10;
//			int xspacing = 100;
//			int xo = 30;
//			int maxneurons = 12;
//			
//			int[] yoa = new int[Main.nn.a.length];
//			for(int n = 0; n < yoa.length; n++) {
//				yoa[n] = Main.nn.a[n].length > maxneurons ? (height-maxneurons*(diameter+yspacing))/2:(height-Main.nn.a[n].length*(diameter+yspacing))/2;
//			}
//			
//			g.setFont(new Font("Arial", Font.BOLD, 9));
//			g.setColor(new Color(0xb0, 0xc0, 0x20));
//			g.fillRect(0, 0, width, height);
//			
//			//Weights
//			for(int n = 0; n < Main.nn.w.length; n++) {
//				for(int i = 0; i < Math.min(maxneurons,Main.nn.w[n][0].length); i++) {
//					for(int j = 0; j < Math.min(maxneurons,Main.nn.w[n].length); j++) {
//						double w = Main.nn.w[n][j][i];
//						g.setColor(w>0 ? new Color(0, 0, 0xff, (int) Math.min(0xff,0xff*w)):new Color(0xff, 0, 0, (int) (Math.min(0xff,0xff*-w))));
//						g.drawLine(xspacing*n+xo+diameter/2, yoa[n]+i*(diameter+yspacing)+diameter/2, xo+xspacing*(n+1)+diameter/2, yoa[n+1]+j*(yspacing+diameter)+diameter/2);
//					}
//				}
//			}
//			//Biases
//			for(int n = 0; n < Main.nn.b.length; n++) {
//				for(int i = 0; i < Math.min(maxneurons,Main.nn.b[n].length); i++) {
//					double wb = Main.nn.b[n][i];
//					g.setColor(wb>0 ? new Color(0, 0xff, 0, (int) Math.min(0xff,0xff*wb)):new Color(0xff, 0x70, 0, (int) (Math.min(0xff,0xff*-wb))));
//					g.drawLine(xspacing*n+xo+diameter/2, height-yoa[n]+diameter/2, xspacing*(n+1)+xo+diameter/2, yoa[n+1]+i*(diameter+yspacing)+diameter/2);
//				}				
//			}
//			//Neurons
//			for(int n = 0; n < Main.nn.a.length; n++) {
//				for(int i = 0; i < Math.min(maxneurons,Main.nn.a[n].length); i++) {
//					double a = Main.nn.a[n][i];
//					int col = (int) Math.abs(Math.min(0xff,0xff*a));
//					g.setColor(new Color(col,col,col));
//					int y = yoa[n]+i*(diameter+yspacing);
//					g.fillOval(xspacing*n+xo, y, diameter, diameter);
//					g.setColor(a>0.8 ? Color.black:Color.white);
//					g.drawString(new DecimalFormat("#.##").format(a)+"", xspacing*n+xo+diameter/4-3, y+diameter/4*3);
//				}
//			}
//			for(int n = 0; n < Main.nn.b.length; n++) {
//				g.setColor(Color.white);
//				g.fillOval(xspacing*n+xo, height-yoa[n], diameter, diameter);
//			}
			Main.nn.render(g, 30, 30, height, width);
			
			int[] pixels = new int[28*28];
			for(int i = 0; i < pixels.length; i++) {
				pixels[i] = (int) (Main.nn.a[0][i]*255*0x10101);
			}
			BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
			image.setRGB(0, 0, 28, 28, pixels, 0, 28);
			g.drawImage(image, 100*Main.nn.a.length, 150, null);
			
			g.drawString((double)Main.nn.correct/(double)Main.nn.total*100+"% correct", 100*Main.nn.a.length, 200);
			if(Main.testing) {
				g.setColor(Color.red);
				g.drawString("Testing...", 100*Main.nn.a.length, 250);				
			}
			if(paint != null)paint.draw(g);
			
			try {
				Thread.sleep(16);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			repaint();
		}
	}
	
	public Window(int width, int height) {
		super("Neural Network by MrCamoga");
		this.height = height;
		this.width = width;
		
		setSize(width, height);
		setResizable(true);
		
		JMenuBar menuBar = new JMenuBar();
		JMenu file = new JMenu("File");
		JMenuItem newnn = new JMenuItem("New NN");
		JMenuItem open = new JMenuItem("Open");
		JMenuItem save = new JMenuItem("Save");
		JMenuItem learning = new JMenuItem("Update lambda");
		newnn.addActionListener(this);
		open.addActionListener(this);
		save.addActionListener(this);
		learning.addActionListener(this);
		
		file.add(newnn);
		file.add(open);
		file.add(save);
		file.add(learning);
		menuBar.add(file);
		
		add(menuBar, BorderLayout.NORTH);
		
		JPanel panel = new Panel();
		add(panel);
		
//		paint = new Paint();
//		panel.addMouseListener(paint);
//		panel.addMouseMotionListener(paint);
		
		setVisible(true);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

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
				//TODO activations
				Main.nn = new NeuralNetwork(new int[] {Integer.parseInt(i.getText()), Integer.parseInt(h.getText()), Integer.parseInt(h1.getText()), Integer.parseInt(o.getText())}, new int[3]);
			}
			break;
		case "Open":
			openState();
			break;
		case "Save":
			saveState();
			break;
		case "Update lambda":
			JPanel panel2 = new JPanel(new GridLayout(1, 2));
			panel2.add(new JLabel("Learning rate (lambda"));
			JTextField l = new JTextField();
			panel2.add(l);
			int result2 = JOptionPane.showConfirmDialog(this, panel2, "Update Learning Rate", JOptionPane.OK_CANCEL_OPTION);
			if(result2 == JOptionPane.OK_OPTION) Main.nn.lambda = Integer.parseInt(l.getText());
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
			Main.nn = new NeuralNetwork(new BufferedInputStream(new FileInputStream(file.getSelectedFile())));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void saveState() {
		try {
			JFileChooser file = new JFileChooser("C:/Users/Yolanda1/Desktop/workspace/Neural Network/");
			int state = file.showSaveDialog(this);
			if(state == JFileChooser.CANCEL_OPTION) return;
			String path = file.getSelectedFile().getAbsolutePath();
			
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(path));
			bos.write(Main.nn.a.length);
			//NN Layers
			for(int n = 0; n < Main.nn.a.length; n++) {
				byte[] bytes = new byte[4];
				ByteBuffer.wrap(bytes).putInt(Main.nn.a[n].length);
				bos.write(bytes);
			}
			for(int n = 0; n < Main.nn.w.length; n++) {
				bos.write(Main.nn.f[n].getId());
			}
			//NN weights
			for(int i = 0; i < Main.nn.w.length; i++) {
				for(int j = 0; j < Main.nn.w[i].length; j++) {
					for(int k = 0; k < Main.nn.w[i][j].length; k++) {
						byte[] bytes = new byte[8];
						ByteBuffer.wrap(bytes).putDouble(Main.nn.w[i][j][k]);
						bos.write(bytes);
						System.err.println(Arrays.toString(bytes));
					}
				}
			}
			//NN Biases
			for(int i = 0; i < Main.nn.b.length; i++) {
				for(int j = 0; j < Main.nn.b[i].length; j++) {
					byte[] bytes = new byte[8];
					ByteBuffer.wrap(bytes).putDouble(Main.nn.b[i][j]);
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
}