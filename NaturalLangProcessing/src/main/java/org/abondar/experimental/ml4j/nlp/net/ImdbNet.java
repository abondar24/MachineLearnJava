package org.abondar.experimental.ml4j.nlp.net;


import org.abondar.experimental.ml4j.utils.FileDownloadUtil;
import org.abondar.experimental.ml4j.utils.UnarchiveUtil;
import org.codehaus.plexus.archiver.tar.TarGZipUnArchiver;
import org.codehaus.plexus.logging.console.ConsoleLoggerManager;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

import static org.abondar.experimental.ml4j.nlp.net.NetConstants.DATASET_URL;
import static org.abondar.experimental.ml4j.nlp.net.NetConstants.IMDB_TAR_GZ_PATH;
import static org.abondar.experimental.ml4j.nlp.net.NetConstants.MAX_SENTENCE_LEN;
import static org.abondar.experimental.ml4j.nlp.net.NetConstants.RANGE;
import static org.abondar.experimental.ml4j.nlp.net.NetConstants.TEST_PATH;
import static org.abondar.experimental.ml4j.nlp.net.NetConstants.TRAIN_PATH;
import static org.abondar.experimental.ml4j.nlp.net.NetConstants.VECTOR_GZ_PATH;
import static org.abondar.experimental.ml4j.nlp.net.NetConstants.VECTOR_URL;

public class ImdbNet {

    private static final Logger logger = LoggerFactory.getLogger(ImdbNet.class);



    private WordVectors loadVectors() {
        logger.info("Load word vectors");
        var vector = new File(VECTOR_GZ_PATH);
        return WordVectorSerializer.loadStaticModel(vector);
    }

    private DataSetIterator getDatasetIterator(boolean training, WordVectors vectors) {
        var path = training ? TRAIN_PATH : TEST_PATH;

        var positivePath = path + "pos";
        var negativePath = path + "neg";

        var positiveFile = new File(positivePath);
        var negativeFile = new File(negativePath);

        Map<String, List<File>> reviewFiles = new HashMap<>();
        reviewFiles.put("Positive", List.of(positiveFile.listFiles()));
        reviewFiles.put("Negative", List.of(negativeFile.listFiles()));

        var sentenceProvider = new FileLabeledSentenceProvider(reviewFiles, RANGE);

        return buildIterator(sentenceProvider, vectors);
    }

    private DataSetIterator buildIterator(FileLabeledSentenceProvider sentenceProvider, WordVectors vectors) {

        return new CnnSentenceDataSetIterator.Builder(CnnSentenceDataSetIterator.Format.CNN2D)
                .sentenceProvider(sentenceProvider)
                .wordVectors(vectors)
                .minibatchSize(32)
                .maxSentenceLength(MAX_SENTENCE_LEN)
                .useNormalizedWordVectors(false)
                .build();
    }

    private ComputationGraphConfiguration buildConfig() {
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        var vectorSize = 300;
        int cnnLayerFeatureMaps = 100;

        var layers = List.of("cnn3", "cnn4", "cnn5");

        return new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.01))
                .convolutionMode(ConvolutionMode.Same)
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer(layers.get(0), buildConvLayer(3, vectorSize, cnnLayerFeatureMaps), "input")
                .addLayer(layers.get(1), buildConvLayer(4, vectorSize, cnnLayerFeatureMaps), "input")
                .addLayer(layers.get(2), buildConvLayer(4, vectorSize, cnnLayerFeatureMaps), "input")
                .addVertex("merge", new MergeVertex(), layers.get(0), layers.get(1), layers.get(2))
                .addLayer("globalPool", buildGlobalPoolingLayer(), "merge")
                .addLayer("out", buildOutputLayer(), "globalPool")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(MAX_SENTENCE_LEN, vectorSize, 1))
                .build();

    }

    private ConvolutionLayer buildConvLayer(int kerSize, int vectorSize, int featureMaps) {
        return new ConvolutionLayer.Builder()
                .kernelSize(kerSize, vectorSize)
                .stride(1, vectorSize)
                .nOut(featureMaps)
                .build();
    }

    private GlobalPoolingLayer buildGlobalPoolingLayer() {
        return new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build();
    }

    private OutputLayer buildOutputLayer() {
        return new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nOut(2)
                .build();
    }

    private ComputationGraph trainNetwork(ComputationGraphConfiguration netConf, DataSetIterator train, DataSetIterator test) {
        logger.info("Init and train network");

        var net = new ComputationGraph(netConf);
        net.init();

        int epochs = 1;

        for (int i = 0; i < epochs; i++) {
            net.fit(train);

            logger.info(String.format("Epoch %d complete. Starting evaluation", i));

            var eval = net.evaluate(test);
            logger.info(eval.stats());
        }

        return net;
    }

    public ComputationGraph buildModel() throws IOException, InterruptedException {
        var imdbArchive = new File(IMDB_TAR_GZ_PATH);

        var executor = Executors.newFixedThreadPool(2);
        if (!imdbArchive.exists()) {
            var imdbThread = new Thread(() -> {
                try {
                    logger.info("Downloading IMDB dataset");
                    FileDownloadUtil.downloadArchive(DATASET_URL, IMDB_TAR_GZ_PATH);
                    UnarchiveUtil.unarchiveTar(imdbArchive);
                } catch (IOException ex) {
                    logger.error(ex.getMessage());
                    System.exit(2);
                }

            });
            executor.execute(imdbThread);
        }

        var vectorArchive = new File(VECTOR_GZ_PATH);
        if (!vectorArchive.exists()) {
            var vectorThread = new Thread(() -> {
                try {
                    logger.info("Downloading Google News Vectors");
                    FileDownloadUtil.downloadArchive(VECTOR_URL, VECTOR_GZ_PATH);
                } catch (IOException ex) {
                    logger.error(ex.getMessage());
                    System.exit(2);
                }
            });
            executor.execute(vectorThread);
        }

        executor.shutdown();
        while (!executor.isTerminated()) {

        }
        logger.info("Downloaded all required data");

        var vectors = loadVectors();

        var trainData = getDatasetIterator(true, vectors);
        var testData = getDatasetIterator(false, vectors);

        var netConfig = buildConfig();

        return trainNetwork(netConfig, trainData, testData);

    }

    public void makeSentencePrediction(String sentencePath, ComputationGraph net) throws IOException {
        var path = Path.of(sentencePath);
        var sentence = Files.readString(path, StandardCharsets.UTF_8);

        var vectors = loadVectors();
        var iterator = getDatasetIterator(false, vectors);
        var features = ((CnnSentenceDataSetIterator) iterator).loadSingleSentence(sentence);

        var predictions = net.outputSingle(features);
        List<String> labels = iterator.getLabels();

        logger.info(String.format("Predictions for %s", sentence));

        for (int i = 0; i < labels.size(); i++) {
            var msg = String.format("P(%s) = %f", labels.get(i), predictions.getDouble(i));
            logger.info(msg);
        }

    }
}
