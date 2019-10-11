package com.camoga.utils;

import java.util.Arrays;


public class PCA {
	
	public static MatrixNd pca(VecNd[] data) {
		double[][] covarianceMatrix = covarianceMatrix(data);
//		System.out.println(Arrays.toString(covarianceMatrix));
		double[][] ev = eigenvectors(covarianceMatrix);
		return new MatrixNd(ev);
	}
	
	public static double[][] eigenvectors(double[][] A) {
		double[][][] QR = QRfact(A);
		double[][] X = null;
		for(int i = 0; i < 1000; i++) {
			X = new MatrixNd(A).multiply(QR[0]).matrix;		
			QR = QRfact(X);
			System.out.println(Arrays.toString(X[0]));
		}
		return X;
	}
	
	public static double[][][] QRfact(double[][] A) {		
		VecNd[] a = new VecNd[A[0].length];
		for(int i  = 0; i < a.length; i++) {
			a[i] = new VecNd(A, i);
		}

		VecNd[] e = new VecNd[A[0].length];
		VecNd[] u = new VecNd[A[0].length];
		
		for(int i = 0; i < u.length; i++) {
			u[i] = a[i].clone();
			for(int j = 0; j < i; j++) {
				u[i].sub(VecNd.mul(e[j],VecNd.dot(a[i], e[j])));
			}
			e[i] = VecNd.normalize(u[i]);
		}
		
		
		MatrixNd Q = new MatrixNd(e);
		
		double[][] r = new double[A.length][A[0].length];
		
		for(int i = 0; i < r.length; i++) {
			for(int j = i; j < r[i].length; j++) {
				r[i][j] = VecNd.dot(a[j], e[i]);
			}
		}
		
		return new double[][][] {Q.matrix, r};
	}
	
	public static double[][] covarianceMatrix(VecNd data[]) {
		int N = data[0].xs.length;
		double[][] covarianceMatrix = new double[N][N];
		
		double[] mean = new double[N];
		
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < data.length; j++) {
				mean[i] += data[j].xs[i];
			}
			mean[i] /= data.length;
		}
		
		for(int i = 0; i < N; i++) {
			for(int j = i; j < N; j++) {
				for(int k = 0; k < data.length; k++) {
					covarianceMatrix[i][j] += (data[k].xs[i]-mean[i])*(data[k].xs[j]-mean[j]);				
				}
				covarianceMatrix[i][j] /= data.length;
			}
		}
		for(int j = 1; j < N; j++) {
			for(int i = 0; i < j; i++) {
				covarianceMatrix[j][i] = covarianceMatrix[i][j];
			}
		}
		return covarianceMatrix;
	}
}