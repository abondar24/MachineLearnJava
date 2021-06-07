package org.abondar.experimental.ml4j.convnet.net;

import org.datavec.api.split.InputSplit;

public class DataWrapper {

    private final int numLabels;
    private final InputSplit trainData;
    private final InputSplit testData;

    public DataWrapper(int numLabels, InputSplit trainData, InputSplit testData) {
        this.numLabels = numLabels;
        this.trainData = trainData;
        this.testData = testData;
    }

    public int getNumLabels() {
        return numLabels;
    }

    public InputSplit getTrainData() {
        return trainData;
    }

    public InputSplit getTestData() {
        return testData;
    }
}
