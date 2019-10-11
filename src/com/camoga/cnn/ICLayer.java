package com.camoga.cnn;

public interface ICLayer {

	public double[][][] forward(double[][][] prev);
	public double[][][] backprop(ICLayer prev, double[][][] cost);
	
	public double[][][] output();

	public int depth();
	public int size();
}
