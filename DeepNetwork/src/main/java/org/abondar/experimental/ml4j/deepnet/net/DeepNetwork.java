package org.abondar.experimental.ml4j.deepnet.net;

import org.abondar.experimental.ml4j.deepnet.columns.Countries;
import org.abondar.experimental.ml4j.deepnet.columns.Genders;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Age;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Balance;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.CreditScore;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.CustomerId;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.EstimatedSalary;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Exited;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Gender;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Geography;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.HasCrCard;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.IsActiveMember;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.NumOfProducts;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.RowNumber;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Surname;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Tenure;

public class DeepNetwork {

    private static final Slf4jLogger LOGGER = new Slf4jLogger(DeepNetwork.class);

    private RecordReader initReader(String filepath) throws IOException, InterruptedException {
        var csvSkipLine = 1;
        var csvDelim = ',';

        var reader = new CSVRecordReader(csvSkipLine, csvDelim);
        var file = new File(filepath);
        var split = new FileSplit(file);

        reader.initialize(split);

        return reader;
    }

    private Schema buildSchema() {
        return new Schema.Builder()
                .addColumnString(RowNumber.name())
                .addColumnInteger(CustomerId.name())
                .addColumnString(Surname.name())
                .addColumnInteger(CreditScore.name())
                .addColumnCategorical(Geography.name(), Countries.getList())
                .addColumnCategorical(Gender.name(), Genders.getList())
                .addColumnsInteger(Age.name(), Tenure.name())
                .addColumnDouble(Balance.name())
                .addColumnsInteger(NumOfProducts.name(), HasCrCard.name(), IsActiveMember.name())
                .addColumnDouble(EstimatedSalary.name())
                .addColumnInteger(Exited.name())
                .build();

    }

    private RecordReader applyTransform(Schema schema, RecordReader recordReader) {
        var transformProcess = new TransformProcess.Builder(schema)
                .removeColumns(RowNumber.name(), CustomerId.name(), Surname.name())
                .categoricalToInteger(Gender.name())
                .categoricalToOneHot(Geography.name())
                .removeColumns("Geography[France]")
                .build();

        return new TransformProcessRecordReader(recordReader, transformProcess);
    }

    private DataSetIterator getDataset(RecordReader reader) {
        var batchSize = 8;
        var labelIndex = 11;
        var numClasses = 2;

        return new RecordReaderDataSetIterator.Builder(reader, batchSize)
                .classification(labelIndex, numClasses)
                .build();
    }

    private NormalizerStandardize initNormalizer(DataSetIterator dataSetIterator){
        var normalizer = new NormalizerStandardize();
        normalizer.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(normalizer);

        return normalizer;
    }

    private  DataSetIteratorSplitter splitToBatches(DataSetIterator dataSetIterator){
        var totalBatches = 1250;
        var ratio = 0.8;
        return new DataSetIteratorSplitter(dataSetIterator, totalBatches, ratio);
    }


    private DenseLayer buildDenseLayer(int in, int out) {
        var dropout = 0.9;
        return new DenseLayer.Builder()
                .nIn(in)
                .nOut(out)
                .activation(Activation.RELU)
                .dropOut(dropout)
                .build();
    }

    private OutputLayer buildOutputLayer(int in, int out) {
        var weightsArray = Nd4j.create(new double[]{0.57, 0.75});
        var loss = new LossMCXENT(weightsArray);

        return new OutputLayer.Builder(loss)
                .nIn(in)
                .nOut(out)
                .activation(Activation.SOFTMAX)
                .build();
    }

    private MultiLayerConfiguration buildNetConfig() {
        var adamLearnRate = 0.015D;

        return new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .updater(new Adam(adamLearnRate))
                .list()
                .layer(buildDenseLayer(11, 6))
                .layer(buildDenseLayer(6, 6))
                .layer(buildDenseLayer(6, 4))
                .layer(buildOutputLayer(4, 2))
                .build();

    }

    private MultiLayerNetwork initTraining(DataSetIteratorSplitter dataset, MultiLayerConfiguration netConfig) {
        var ui = UIServer.getInstance();
        var statsStorage = new InMemoryStatsStorage();

        LOGGER.info("Init training");
        var network = new MultiLayerNetwork(netConfig);
        var iterations = 100;

        network.init();
        network.setListeners(new ScoreIterationListener(iterations),
                new StatsListener(statsStorage));

        ui.attach(statsStorage);

        //train set
        var epochs = 100;
        network.fit(dataset.getTrainIterator(),epochs);

        var labels = List.of("0","1");
        var eval = network.evaluate(dataset.getTestIterator(), labels);
        LOGGER.info(eval.stats());

        return network;

    }

    private void writeModel (MultiLayerNetwork network,NormalizerStandardize normalizer) throws IOException{
        var file = new File("model.zip");
        ModelSerializer.writeModel(network,file,true);
        ModelSerializer.addNormalizerToModel(file,normalizer);
    }

    public void buildModel(String filePath) throws IOException, InterruptedException {
        var reader = initReader(filePath);
        var schema = buildSchema();

        var transformReader = applyTransform(schema, reader);
        var dataset = getDataset(transformReader);
        var normalizer = initNormalizer(dataset);
        var splitDataset = splitToBatches(dataset);

        var netConfig = buildNetConfig();

        var net = initTraining(splitDataset,netConfig);
        writeModel(net,normalizer);
    }


}
