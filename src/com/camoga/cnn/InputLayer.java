package com.camoga.cnn;

import java.util.Arrays;

public class InputLayer implements ICLayer {

	private double[][][] input;
	
	public InputLayer(int channels, int size) {
		input = new double[channels][size][size];
	}

	public double[][][] forward(double[][][] input) {
		this.input = input;
		return input;
	}
	
	public double[][][] backprop(ICLayer layer, double[][][] prev) {
		return null;
	}

	public int depth() {
		return input.length;
	}
	
	public int size() {
		return input[0].length;
	}

	public double[][][] output() {
		return input;
	}
}