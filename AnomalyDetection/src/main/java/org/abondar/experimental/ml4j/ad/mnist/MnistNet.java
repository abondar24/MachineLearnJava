package org.abondar.experimental.ml4j.ad.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class MnistNet {
    private static final Logger logger = LoggerFactory.getLogger(MnistNet.class);


    public void runComputations() {
        try {
            var data = extractData();
            var net = trainNetwork(data.get(0));

            evaluateNetwork(net, data);
        } catch (IOException ex) {
            logger.error(ex.getMessage());
            System.exit(2);
        }
    }


    private List<List<INDArray>> extractData() throws IOException {
        List<List<INDArray>> dataset = new ArrayList<>();

        var batch = 100;
        var examples = 50000;
        var iterator = new MnistDataSetIterator(batch, examples, false);

        List<INDArray> trainFeatures = new ArrayList<>();
        List<INDArray> testFeatures = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();

        var r = new Random(12345);
        var splitHoldout = 80;
        var dimension = 1;
        while (iterator.hasNext()) {
            var ds = iterator.next();
            var split = ds.splitTestAndTrain(splitHoldout, r);
            trainFeatures.add(split.getTrain().getFeatures());

            var dsTst = split.getTest();
            testFeatures.add(dsTst.getFeatures());
            var indexes = Nd4j.argMax(dsTst.getLabels(), dimension);
            testLabels.add(indexes);
        }

        dataset.add(trainFeatures);
        dataset.add(testFeatures);
        dataset.add(testLabels);
        return dataset;
    }


    private MultiLayerConfiguration configureNet() {
        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaGrad(0.05))
                .activation(Activation.RELU)
                .l2(0.0001)
                .list()
                .layer(buildDenseLayer(784, 250))
                .layer(buildDenseLayer(250, 10))
                .layer(buildDenseLayer(10, 250))
                .layer(buildOutputLayer(250, 784))
                .build();
    }

    private DenseLayer buildDenseLayer(int nIn, int nOut) {
        return new DenseLayer.Builder()
                .nIn(nIn)
                .nOut(nOut)
                .build();
    }

    private OutputLayer buildOutputLayer(int nIn, int nOut) {
        return new OutputLayer.Builder()
                .nIn(nIn)
                .nOut(nOut)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .build();
    }

    private MultiLayerNetwork trainNetwork(List<INDArray> trainFeatures) {
        var printIters = 1;
        var epochs = 30;

        var net = new MultiLayerNetwork(configureNet());
        net.setListeners(new ScoreIterationListener(printIters));

        logger.info("Start training");
        for (var i = 0; i < epochs; i++) {
            logger.info(String.format("Starting epoch %d", i));

            trainFeatures.forEach(tf -> net.fit(tf, tf));
        }

        return net;
    }

    private void evaluateNetwork(MultiLayerNetwork network, List<List<INDArray>> data) {
        Map<Integer, List<Pair<Double, INDArray>>> digitLists = new HashMap<>();

        var testFeatures = data.get(1);
        var testLabels = data.get(2);

        for (var i = 0; i < 10; i++) {
            digitLists.put(i, new ArrayList<>());
        }

        for (var i = 0; i < testFeatures.size(); i++) {
            var testData = testFeatures.get(i);
            var labels = testLabels.get(i);
            var rows = testData.rows();

            for (var j = 0; j < rows; j++) {
                var example = testData.getRow(j, true);
                var digit = (int) labels.getDouble(j);
                var score = network.score(new DataSet(example, example));

                var allPairsDigit = digitLists.get(digit);
                allPairsDigit.add(new Pair<>(score, example));
            }
        }

        digitLists.forEach((k, v) -> v.sort(getComparator()));

        List<INDArray> bestScores = new ArrayList<>(50);
        List<INDArray> worstScores = new ArrayList<>(50);

        for (var i = 0; i < 10; i++) {
            var list = digitLists.get(i);
            for (var j = 0; j < 5; j++) {
                bestScores.add(list.get(j).getRight());
                worstScores.add(list.get(list.size() - j - 1).getRight());
            }
        }

        visualizeData(bestScores, "Best (Low Rec. Error)");
        visualizeData(worstScores, "Worst (High Rec. Error)");

    }

    private Comparator<Pair<Double, INDArray>> getComparator() {
        return Comparator.comparingDouble(Pair::getLeft);
    }

    private void visualizeData(List<INDArray> scores, String title) {
        var imgScale = 2.0;
        var vis = new MnistVisualiser(imgScale, scores, title);
        vis.visualise();
    }
}
