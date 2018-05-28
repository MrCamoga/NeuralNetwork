package com.camoga.nn;

/**
 * 
 *	SIGMOID(0) <br>
 *	TANH(1)
 */
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