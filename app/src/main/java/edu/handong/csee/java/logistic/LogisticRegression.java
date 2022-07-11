package edu.handong.csee.java.logistic;

import java.util.ArrayList;

public class LogisticRegression {

    public static void main(String[] args) {
        ArrayList<DataInstance> trainInstances = new ArrayList<>();
        ArrayList<DataInstance> testInstances = new ArrayList<>();

        // make instances for train data and test data
        trainInstances.add(new DataInstance(new double[]{1.0, 2.0}, 0));
        trainInstances.add(new DataInstance(new double[]{2.0, 3.0}, 0));
        trainInstances.add(new DataInstance(new double[]{3.0, 1.0}, 0));
        trainInstances.add(new DataInstance(new double[]{4.0, 3.0}, 1));
        trainInstances.add(new DataInstance(new double[]{5.0, 3.0}, 1));
        trainInstances.add(new DataInstance(new double[]{6.0, 2.0}, 1));

        testInstances.add(new DataInstance(new double[]{5.0, 2.0}, 1));

        Model model = new Model(2001, 0.01, 0.00001);

        // train and test
        model.train(trainInstances);
        model.test(testInstances);
    }
}
