package edu.handong.csee.java.logistic;

import java.util.ArrayList;

public class Model {
    private int epoch;
    private double learningRate;
    private double initCost;
    private double computedCost;
    private double th;
    private Classifier classifier;

    public Model(int epoch, double learningRate, double th) {
        this.epoch = epoch;
        this.learningRate = learningRate;
        this.th = th;
        this.classifier = new Classifier();
    }

    public void train(ArrayList<DataInstance> instances){
        classifier.setWeightsBias(instances.get(0).getInputSize());
        initCost = classifier.costFunction(instances);

        for (int i = 0; i < epoch; i ++){
            classifier.gradientDescent(instances, learningRate);

            computedCost = classifier.costFunction(instances);

            if (Math.abs(computedCost - initCost) < th) break;
            initCost = computedCost;

            if (i % 100 == 0) {
                System.out.printf("epoch: %d, Loss: %f\n", i, classifier.costFunction(instances));
            }
        }

    }

    public void test(ArrayList<DataInstance> instances){
        int count = 0 ;

        for (DataInstance instance : instances){
            if (instance.getLabel() == classifier.classify(instance.getInputs(), 0.5)) {
                count++;
            }
        }

        System.out.printf("Test-set Accuracy: %d\n", count / instances.size() * 100);
    }
}
