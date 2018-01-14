package com.camoga.nn;

public enum ActivationFunctions {
	SIGMOID(0), TANH(1);
	
	private int id;
	
	ActivationFunctions(int id) {
		this.id = id;
	}
	
	public int getId() {
		return id;
	}
}
