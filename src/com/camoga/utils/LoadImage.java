package com.camoga.utils;

import java.io.BufferedInputStream;
import java.io.IOException;

import com.camoga.examples.autoencoder.Autoencoder;

public class LoadImage {
	
	public double[] pgm(String path) throws IOException {
		double[] IMAGE = null;
		
		BufferedInputStream imageBuffer = new BufferedInputStream(getClass().getResourceAsStream(path));
		try {
			byte[] data = new byte[4];
			imageBuffer.read(data, 0, 2);
			imageBuffer.skip(1);
			String pgmtype = new String(data).trim();
			int checkName = imageBuffer.read();
			if(checkName==0x23) {
				System.out.println("name");
				while(imageBuffer.read() != 0x0A) {}
				checkName = imageBuffer.read();
			}
			if(pgmtype.equals("P5")) {
				data = new byte[4];
				data[0] = (byte) checkName;
				boolean blank = false;
				for(int i = 1; !blank; i++) {
					int d = imageBuffer.read();
					if(d==0x20) blank = true;
					else data[i] = (byte) d;
				}
				Autoencoder.WIDTH = Integer.parseInt(new String(data).trim());
				
				
				data = new byte[4];
				blank = false;
				for(int i = 0; !blank; i++) {
					int d = imageBuffer.read();
					if(d==0x0A) blank = true;
					else data[i] = (byte) d;
				}
				Autoencoder.HEIGHT = Integer.parseInt(new String(data).trim());
				IMAGE = new double[Autoencoder.WIDTH*Autoencoder.HEIGHT];
				
				data = new byte[3];
				imageBuffer.read(data, 0, 3);
				
				
				double maximumValue = Double.parseDouble(new String(data).trim());
				imageBuffer.skip(1);
				
				for(int i = 0; i < Autoencoder.WIDTH*Autoencoder.HEIGHT; i++) {
					IMAGE[i] = imageBuffer.read()/maximumValue;
				}
			} else if(pgmtype.equals("P2")) {
				boolean blank = false;
				data = new byte[4];
				data[0] = (byte) checkName;
				for(int i = 1; !blank; i++) {
					int d = imageBuffer.read();
					if(d==0x20) blank = true;
					else data[i] = (byte) d;
				}
				Autoencoder.WIDTH = Integer.parseInt(new String(data).trim());
				
				blank = false;
				for(int i = 0; !blank; i++) {
					int d = imageBuffer.read();
					if(d==0x0A) blank = true;
					else data[i] = (byte) d;
				}
				Autoencoder.HEIGHT = Integer.parseInt(new String(data).trim());
				IMAGE = new double[Autoencoder.WIDTH*Autoencoder.HEIGHT];
				
				imageBuffer.read(data, 0, 3);
				double maximumValue = Double.parseDouble(new String(data).trim());
				imageBuffer.skip(1);
				for(int i = 0; i < Autoencoder.WIDTH*Autoencoder.HEIGHT; i++) {
					data = new byte[4];
					blank = false;
					for(int j = 0; !blank; j++) {
						int d = imageBuffer.read();
						if(d==0x20 || d==0x0A) blank = true;
						else data[j] = (byte) d;
					}
					int grayValue = Integer.parseInt(new String(data).trim());
					IMAGE[i] = grayValue/maximumValue;
				}
			}
			
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		if(IMAGE == null) throw new IOException();
		return IMAGE;
	}
}
