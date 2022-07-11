package edu.handong.csee.java.logistic;

public class DataInstance {
    private double[] inputs;
    private int label;

    DataInstance(double[] x, int y){
        this.inputs = x;
        this.label = y;
    }

    public double[] getInputs(){
        return this.inputs;
    }

    public double getAInput(int idx){
        return this.inputs[idx];
    }

    public int getLabel(){
        return this.label;
    }

    public int getInputSize(){
        return inputs.length;
    }

}
