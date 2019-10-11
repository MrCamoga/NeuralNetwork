package com.camoga.utils;

public class VecNd {
	
	public double[] xs;
	
	public VecNd() {
		this(0, 0, 0);
	}
	
	public VecNd(double[][] A, int column) {
		xs = new double[A.length];
		for(int i = 0; i < xs.length; i++) {
			xs[i] = A[i][column];
		}
	}
	
	/**
	 * 
	 * @param x DO NOT INCLUDE HOMOGENEOUS COORDINATE IN ARRAY. INSTEAD, SET IT MANUALLY IN w0
	 */
	public VecNd(double...x) {
		this.xs = x;
	}
	
	public VecNd sub(VecNd v) {
		for(int i = 0; i < dim(); i++) {
			xs[i] -= v.xs[i];
		}
		return this;
	}
	
	public static VecNd mul(VecNd v, double s) {
		double[] ys = new double[v.dim()];
		for(int i = 0; i < ys.length; i++) {
			ys[i] = v.xs[i]*s;
		}
		
		return new VecNd(ys);
	}
	
	/**
	 * 
	 * @return modulus of vector
	 */
	public double mod() {
		double result = 0;
		for(double x : xs) result += x*x;
		
		return Math.sqrt(result);
	}
	
	public static VecNd normalize(VecNd v) {
		double[] ys = new double[v.dim()];
		double mod = v.mod();
		for(int i = 0; i < ys.length; i++) {
			ys[i] = v.xs[i]/mod;
		}
		return new VecNd(ys);
	}
	
	public int dim() {
		return xs.length;
	}
	
	public static double dot(VecNd v, VecNd w) {
		if(v.dim() != w.dim()) throw new RuntimeException("Vector dimensions are different");
		double sum = 0;
		for(int i = 0; i < v.dim(); i++) {
			sum += v.xs[i]*w.xs[i];
		}
		return sum;
	}
	
	public VecNd clone() {
		return new VecNd(xs.clone());		
	}
}
