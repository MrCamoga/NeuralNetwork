package com.camoga.cnn;

public interface INLayer {

	public double[] forward(double[] prev);
	public double[] backprop(INLayer prev, double[] cost);
	
	public double[] output();
	
	public int size();
}