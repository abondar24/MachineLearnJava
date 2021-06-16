package org.abondar.experimental.ml4j.lstm.net;

import org.abondar.experimental.ml4j.utils.FileDownloadUtil;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class UciNet {

    private static final Logger logger = LoggerFactory.getLogger(UciNet.class);

    private static final String DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/";

    private static final String DATA_FILE = "synthetic_control.data";

    private static final String FEATURES_DIR = "/features";

    private static final String LABELS_DIR = "/labels";

    private static final String DATA_DIR = "data/uci/";

    private static final String DATA_TRAIN_DIR = DATA_DIR + "train";

    private static final String DATA_TRAIN_FEATURES = DATA_TRAIN_DIR + FEATURES_DIR;

    private static final String DATA_TRAIN_LABELS = DATA_TRAIN_DIR + LABELS_DIR;

    private static final String DATA_TEST_DIR = DATA_DIR + "test";

    private static final String DATA_TEST_FEATURES = DATA_TEST_DIR + FEATURES_DIR;

    private static final String DATA_TEST_LABELS = DATA_TEST_DIR + LABELS_DIR;

    public void runComputations() {
        try {
            createDirectories();

            var dataPath = DATA_DIR + DATA_FILE;
            var dataFile = new File(dataPath);
            if (!dataFile.exists()) {
                logger.info("Downloading data");
                FileDownloadUtil.downloadFile(DATA_URL + DATA_FILE, dataPath);
            }

            extractData(dataPath);


            var maxId = 449;
            var trainIterator = convertToIterator(maxId, DATA_TRAIN_FEATURES, DATA_TRAIN_LABELS);

            maxId = 149;
            var testIterator = convertToIterator(maxId, DATA_TEST_FEATURES, DATA_TEST_LABELS);

            var normalization = new NormalizerStandardize();
            normalization.fit(trainIterator);

            trainIterator.setPreProcessor(normalization);
            testIterator.setPreProcessor(normalization);

            var model = trainModel(trainIterator, testIterator);
            evaluateModel(model, testIterator);

        } catch (IOException | InterruptedException ex) {
            logger.error(ex.getMessage());
            System.exit(2);
        }

    }

    private void createDirectories() {
        var uciDir = new File(DATA_DIR);
        if (!uciDir.exists()) {
            uciDir.mkdirs();
        }

        var testDir = new File(DATA_TEST_DIR);
        var trainDir = new File(DATA_TRAIN_DIR);
        if (!testDir.exists() || !trainDir.exists()) {
            testDir.mkdirs();
            trainDir.mkdirs();

            var trainFeatures = new File(DATA_TRAIN_FEATURES);
            if (!trainFeatures.exists()) {
                trainFeatures.mkdirs();
            }

            var testFeatures = new File(DATA_TEST_FEATURES);
            if (!testFeatures.exists()) {
                testFeatures.mkdirs();
            }

            var trainLabels = new File(DATA_TRAIN_LABELS);
            if (!trainLabels.exists()) {
                trainLabels.mkdirs();
            }

            var testLabels = new File(DATA_TEST_LABELS);
            if (!testLabels.exists()) {
                testLabels.mkdirs();
            }
        }

    }

    private void extractData(String datafile) throws IOException {
        logger.info("Extracting data");

        var data = readFile(datafile);
        var sequences = data.split("\n");

        List<Pair<String, Integer>> featuresLabels = new ArrayList<>();
        var lines = 0;

        for (var seq : sequences) {
            seq = seq.replaceAll("\\s+", "\n");
            featuresLabels.add(new Pair<>(seq, lines++ / 100));
        }

        Collections.shuffle(featuresLabels, new Random(12345));

        saveToCsv(featuresLabels);

    }

    private String readFile(String dataFile) throws IOException {
        Path path = Path.of(dataFile);
        return Files.readString(path);
    }


    private void saveToCsv(List<Pair<String, Integer>> featuresLabels) throws IOException {
        var trainCount = 0;
        var testCount = 0;
        var splitNum = 450;

        File featureFile;
        File labelFile;

        var fileExt = ".csv";

        for (var seqPair : featuresLabels) {
            if (trainCount < splitNum) {
                featureFile = new File(DATA_TRAIN_FEATURES + "/" + trainCount + fileExt);
                labelFile = new File(DATA_TRAIN_LABELS + "/" + trainCount + fileExt);
                trainCount++;
            } else {
                featureFile = new File(DATA_TEST_FEATURES + "/" + testCount + fileExt);
                labelFile = new File(DATA_TEST_LABELS + "/" + testCount + fileExt);
                testCount++;
            }

            writeToFile(featureFile, seqPair.getFirst());
            writeToFile(labelFile, seqPair.getSecond().toString());

        }
    }

    private void writeToFile(File file, String line) throws IOException {
        var path = Path.of(file.getPath());

        if (!file.exists()) {
            Files.writeString(path,line);
        } else {
            Files.write(path,line.getBytes(), StandardOpenOption.APPEND);
        }

    }

    private DataSetIterator convertToIterator(int maxIdInclude,
                                              String featureDir, String labelDir) throws
            IOException, InterruptedException {
        var minIdInclude = 0;
        var miniBatchSize = 10;
        var numLabels = 6;

        var featuresReader = new CSVSequenceRecordReader();
        var featureSplit = new NumberedFileInputSplit(featureDir + "/%d.csv", minIdInclude, maxIdInclude);
        featuresReader.initialize(featureSplit);

        var labelReader = new CSVSequenceRecordReader();
        var labelSplit = new NumberedFileInputSplit(labelDir + "/%d.csv", minIdInclude, maxIdInclude);
        labelReader.initialize(labelSplit);


        return new SequenceRecordReaderDataSetIterator(featuresReader, labelReader, miniBatchSize,
                numLabels, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
    }

    private ComputationGraphConfiguration configureNetwork() {
        return new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(0.5)
                .graphBuilder()
                .addInputs("trainFeatures")
                .setOutputs("predictSequence")
                .addLayer("L1", buildLstmLayer(), "trainFeatures")
                .addLayer("predictSequence", buildOutputLayer(), "L1")
                .build();
    }

    private LSTM buildLstmLayer() {
        return new LSTM.Builder()
                .nIn(1)
                .nOut(10)
                .activation(Activation.TANH)
                .build();
    }

    private RnnOutputLayer buildOutputLayer() {
        return new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(10)
                .nOut(6)
                .build();
    }

    private ComputationGraph trainModel(DataSetIterator trainIterator, DataSetIterator testIterator) {
        var model = new ComputationGraph(configureNetwork());
        model.init();

        logger.info("Start training model.");

        var scoreIterations = 20;
        var scoreListener = new ScoreIterationListener(scoreIterations);

        var evalFreq = 1;
        var evalListener = new EvaluativeListener(testIterator, evalFreq, InvocationType.EPOCH_END);

        model.setListeners(scoreListener, evalListener);

        var epochs = 100;
        model.fit(trainIterator, epochs);

        return model;
    }

    private void evaluateModel(ComputationGraph model, DataSetIterator testIterator) {
        logger.info("Evaluating model");

        var eval = model.evaluate(testIterator);
        logger.info(eval.stats());
    }
}
