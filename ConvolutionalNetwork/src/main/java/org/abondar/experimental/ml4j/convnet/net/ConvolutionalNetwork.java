package org.abondar.experimental.ml4j.convnet.net;

import org.apache.ant.compress.taskdefs.Unzip;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public class ConvolutionalNetwork {

    private final Slf4jLogger logger = new Slf4jLogger(ConvolutionalNetwork.class);

    private final int numChannels = 3;
    private final int batchSize = 10;

    private DataWrapper getData(String filepath, ParentPathLabelGenerator labelGenerator) {
        var zip = new File(filepath);
        var datasetPath = filepath.replace(".zip","");
        var dataset = new File(datasetPath);
        unzipDataset(zip);
        var weights = List.of(80, 20);

        var fileSplit = new FileSplit(dataset, NativeImageLoader.ALLOWED_FORMATS, getRandom());
        int numLabels = Objects.requireNonNull(
                fileSplit.getRootDir().listFiles(File::isDirectory))
                .length;


        var pathFilter = new BalancedPathFilter(getRandom(), NativeImageLoader.ALLOWED_FORMATS, labelGenerator);
        var inputSplits = fileSplit.sample(pathFilter, weights.get(0), weights.get(1));

        return new DataWrapper(numLabels, inputSplits[0], inputSplits[1]);
    }

    private ImageTransform getTransform() {
        var flipSeed = 123;
        var warpDelta = 42;
        var rotateAngle = 40;

        var flipTransform = new FlipImageTransform(getRandom());
        var flipTransform1 = new FlipImageTransform(getRandom(flipSeed));
        var warpTransform = new WarpImageTransform(getRandom(), warpDelta);
        var rotateTransform = new RotateImageTransform(getRandom(), rotateAngle);

        List<Pair<ImageTransform, Double>> pipeline = List.of(
                new Pair<>(flipTransform, 0.7),
                new Pair<>(flipTransform1, 0.6),
                new Pair<>(warpTransform, 0.5),
                new Pair<>(rotateTransform, 0.4)
        );

        return new PipelineImageTransform(pipeline);
    }

    private void unzipDataset(File zip) {
        var unzip = new Unzip();

        unzip.setSrc(zip);
        unzip.setDest(new File("data"));
        unzip.execute();
    }

    private Random getRandom(int... seed) {
        var defaultSeed = 42;
        if (seed.length > 0) {
            return new Random(seed[0]);
        }

        return new Random(defaultSeed);
    }

    private MultiLayerConfiguration configNetwork(int numLabels) {
        var mean = 0.0;
        var std = 0.01;
        var distribution = new NormalDistribution(mean, std);
        var updaterInitVal = 1e-2;
        var biasUpdaterInitVal = 2e-2;
        var l2reg = 5 * 1e-4;

        var layer1 = new ConvLayerSetup(1, 11, 96);
        var layer2 = new ConvLayerSetup(2, 5, 256);

        var height = 30;
        var width = 30;
        var depth = 3;
        var inputType = InputType.convolutional(height, width, depth);

        return new NeuralNetConfiguration.Builder()
                .weightInit(distribution)
                .activation(Activation.RELU)
                .updater(getUpdater(updaterInitVal))
                .biasUpdater(getUpdater(biasUpdaterInitVal))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(l2reg)
                .list()
                .layer(buildConvLayer(layer1.getKernelSize(), layer1.getOutNodes(), numChannels))
                .layer(layer1.getLayerNum(), buildRespNormalization("lrn1"))
                .layer(buildSubSamplingLayer())
                .layer(buildConvLayer(layer2.getKernelSize(), layer2.getOutNodes()))
                .layer(layer2.getLayerNum(), buildRespNormalization("lrn2"))
                .layer(buildSubSamplingLayer())
                .layer(buildDenseLayer())
                .layer(buildDenseLayer())
                .layer(buildOutputLayer(numLabels))
                .setInputType(inputType)
                .backpropType(BackpropType.Standard)
                .build();
    }

    private Nesterovs getUpdater(double initialValue) {
        var momentum = 0.9;
        return new Nesterovs(getStep(initialValue), momentum);
    }

    private StepSchedule getStep(double initialValue) {
        var decay = 0.1;
        var step = 100000;
        return new StepSchedule(ScheduleType.ITERATION, initialValue, decay, step);
    }

    private Layer buildConvLayer(int kernelSize, int nOut, int... nIn) {
        var stride = 1;

        var builder = new ConvolutionLayer.Builder(kernelSize, kernelSize);
        if (nIn.length > 0) {
            builder = builder.nIn(nIn[0]);
        }

        return builder.nOut(nOut)
                .stride(stride, stride)
                .activation(Activation.RELU)
                .build();
    }

    private LocalResponseNormalization buildRespNormalization(String name) {
        return new LocalResponseNormalization.Builder()
                .name(name)
                .build();
    }

    private Layer buildSubSamplingLayer() {
        var kernelSize = 3;
        return new SubsamplingLayer.Builder(PoolingType.MAX)
                .kernelSize(kernelSize, kernelSize)
                .build();
    }

    private Layer buildDenseLayer() {
        var outNodes = 500;
        return new DenseLayer.Builder()
                .nOut(outNodes)
                .activation(Activation.RELU)
                .build();
    }

    private Layer buildOutputLayer(int nOut) {
        return new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(nOut)
                .activation(Activation.SOFTMAX)
                .build();
    }

    private void trainNetwork(DataWrapper data, ImageTransform transform, ParentPathLabelGenerator labelGenerator) throws IOException {
        var imgHeight = 30;
        var imgWidth = 30;
        var scaler = new ImagePreProcessingScaler(0, 1);
        var imageReader = new ImageRecordReader(imgHeight, imgWidth, numChannels, labelGenerator);

        var network = new MultiLayerNetwork(configNetwork(data.getNumLabels()));
        network.init();

        trainWithoutTransforms(scaler, network, imageReader, data);
        trainWithTransforms(scaler, network, imageReader, transform, data);

        saveModel(network, scaler);
    }

    private void saveModel(MultiLayerNetwork network, DataNormalization scaler) throws IOException {
        var file = new File("cnn.zip");
        ModelSerializer.writeModel(network, file, true);
        ModelSerializer.addNormalizerToModel(file, scaler);
    }

    private void trainWithoutTransforms(DataNormalization scaler, MultiLayerNetwork network,
                                        ImageRecordReader imageRecordReader, DataWrapper data) throws IOException {
        var labelIndex = 1;
        imageRecordReader.initialize(data.getTrainData(), null);
        var datasetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize,
                labelIndex, data.getNumLabels());
        scaler.fit(datasetIterator);
        datasetIterator.setPreProcessor(scaler);


        var iterations = 100;
        var epochs = 100;
        var scoreListener = new ScoreIterationListener(iterations);
        network.setListeners(scoreListener);
        network.fit(datasetIterator, epochs);

    }

    private void trainWithTransforms(DataNormalization scaler, MultiLayerNetwork network,
                                     ImageRecordReader imageRecordReader, ImageTransform imageTransform,
                                     DataWrapper data) throws IOException {
        var labelIndex = 1;
        imageRecordReader.initialize(data.getTrainData(), imageTransform);
        var datasetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize,
                labelIndex, data.getNumLabels());
        scaler.fit(datasetIterator);
        datasetIterator.setPreProcessor(scaler);

        var epochs = 100;
        network.fit(datasetIterator, epochs);

        imageRecordReader.initialize(data.getTestData());
        datasetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize,
                labelIndex, data.getNumLabels());
        scaler.fit(datasetIterator);
        datasetIterator.setPreProcessor(scaler);

        var eval = network.evaluate(datasetIterator);
        var msg = String.format("args = [%s]", eval.stats());
        logger.info(msg);
    }

    public void buildModel(String filepath) throws InterruptedException, IOException {
        var labelGenerator = new ParentPathLabelGenerator();
        var data = getData(filepath, labelGenerator);
        var transform = getTransform();
        trainNetwork(data, transform, labelGenerator);
    }
}
