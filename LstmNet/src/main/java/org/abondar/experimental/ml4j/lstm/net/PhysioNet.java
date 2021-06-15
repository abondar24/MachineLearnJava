package org.abondar.experimental.ml4j.lstm.net;

import org.abondar.experimental.ml4j.utils.FileDownloadUtil;
import org.abondar.experimental.ml4j.utils.UnarchiveUtil;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class PhysioNet {

    private static final String DATASET_URL = "https://dl4jdata.blob.core.windows.net/training/physionet2012/";

    private static final String TAR_GZ = "physionet2012.tar.gz";

    private static final String DATA_DIR = "data/";

    private static final String TAR_GZ_PATH = DATA_DIR + TAR_GZ;

    private static final String DATASET_ROOT = DATA_DIR + "physionet2012";

    private static final String FEATURE_DIR = DATASET_ROOT + "/sequence";

    private static final String LABEL_DIR = DATASET_ROOT + "/mortality";

    private static final int RANDOM_SEED = 1234;

    private static final Logger logger = LoggerFactory.getLogger(PhysioNet.class);


    public void runComputations() {
        try {
            var archive = new File(TAR_GZ_PATH);
            if (!archive.exists()) {
                logger.info("Downloading dataset.");
                FileDownloadUtil.downloadFile(DATASET_URL + TAR_GZ, TAR_GZ_PATH);
            }

            var dataset = new File(DATASET_ROOT);
            if (!dataset.exists()) {
                logger.info("Extracting data ");
                UnarchiveUtil.unarchiveTar(archive);
            }

            int minId = 0;
            int maxId = 3199;
            var trainDataset = extractData(minId, maxId);

            minId = 3200;
            maxId = 3999;
            var testDataset = extractData(minId, maxId);

            var netConf = configureNetwork(trainDataset);
            var net = trainNetwork(netConf, trainDataset);
            evaluateData(net,testDataset);

        } catch (IOException | InterruptedException ex) {
            logger.error(ex.getMessage());
            System.exit(2);
        }

    }

    private DataSetIterator extractData(int minIdInclude, int maxIdInclude) throws
            IOException, InterruptedException {
        int skipLines = 1;
        int minibatchSize = 100;
        int numLabels = 2;
        String csvDelim = ",";

        var featuresReader = new CSVSequenceRecordReader(skipLines, csvDelim);
        var featureSplit = new NumberedFileInputSplit(FEATURE_DIR + "/%d.csv", minIdInclude, maxIdInclude);
        featuresReader.initialize(featureSplit);

        var labelReader = new CSVSequenceRecordReader();
        var labelSplit = new NumberedFileInputSplit(LABEL_DIR + "/%d.csv", minIdInclude, maxIdInclude);
        labelReader.initialize(labelSplit);

        return new SequenceRecordReaderDataSetIterator(featuresReader, labelReader, minibatchSize,
                numLabels, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
    }

    private ComputationGraphConfiguration configureNetwork(DataSetIterator trainIterator) {
        return new NeuralNetConfiguration.Builder()
                .seed(RANDOM_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .dropOut(0.9)
                .graphBuilder()
                .addInputs("features")
                .setOutputs("predictMortality")
                .addLayer("L1", buildLstmLayer(), "features")
                .addLayer("predictMortality", buildOutputLayer(), "L1")
                .build();

    }

    private LSTM buildLstmLayer() {
        return new LSTM.Builder()
                .nIn(86)
                .nOut(200)
                .forgetGateBiasInit(1)
                .activation(Activation.TANH)
                .build();
    }

    private RnnOutputLayer buildOutputLayer() {
        return new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(200)
                .nOut(2)
                .build();
    }

    private ComputationGraph trainNetwork(ComputationGraphConfiguration netConf, DataSetIterator trainData) {
        logger.info("Training model");
        var model = new ComputationGraph(netConf);

        var numEpochs = 15;
        for (int i=0;i<numEpochs;i++){
            logger.info(String.format("Running Epoch %d",i+1));
            model.fit(trainData);
        }

        trainData.reset();

        return model;
    }

    private void evaluateData(ComputationGraph model, DataSetIterator testData) {
        logger.info("Evaluating model");
        var thresholdSteps = 100;
        var eval = new ROC(thresholdSteps);

        while (testData.hasNext()) {
            var batch = testData.next();
            var output = model.output(batch.getFeatures());
            eval.evalTimeSeries(batch.getLabels(), output[0]);
        }

        logger.info(String.format("Evaluated AUC: %f",eval.calculateAUC()));
        logger.info(eval.stats());
    }
}
