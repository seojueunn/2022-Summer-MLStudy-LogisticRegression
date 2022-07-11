package edu.handong.csee.java.logistic;

import java.util.ArrayList;

public class Classifier {
    private double[] weights;
    private double bias;

    public void setWeightsBias(int size){
        this.bias = 0.0;
        this.weights = new double[size]; // default value is 0
    }

    public double compHypothesis(double[] inputs){
        double z = 0.0;

        for (int i = 0; i < inputs.length; i ++){
            z += inputs[i] * weights[i];
        }
        z += this.bias;

        return sigmoidFunction(z);
    }

    public double sigmoidFunction(double z){
        return 1 / (1 + Math.exp(-z));
    }

    public double costFunction(ArrayList<DataInstance> instances){
        double loss = 0.0;

        for (DataInstance inst : instances){
            double predict_y = compHypothesis(inst.getInputs());
            int label_y = inst.getLabel();
            loss += label_y * Math.log(predict_y) + (1.0 - label_y) * Math.log(1.0 - predict_y);
        }

        return -loss / instances.size();
    }

    public void gradientDescent(ArrayList<DataInstance> instances, double rate){
        double update_bias = 0.0;
        double[] update_weight = new double[weights.length];

        for (DataInstance instance : instances){
            double dist = instance.getLabel() - compHypothesis(instance.getInputs());

            update_bias += dist;

            for (int i = 0; i < weights.length; i ++) {
                update_weight[i] += dist * instance.getAInput(i);
            }
        }

        this.bias += update_bias * rate / instances.size();

        for (int i = 0; i < weights.length; i ++){
            this.weights[i] += update_weight[i] * rate / instances.size();
        }
    }

    public int classify(double[] inputs, double threshold){
        double p = compHypothesis(inputs);
        if (p >= threshold) return 1;
        else return 0;
    }
}
