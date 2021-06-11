package org.abondar.experimental.ml4j.convnet.net;

import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class ConvolutionalNetworkChecker {
    private static final Logger logger = LoggerFactory.getLogger(ConvolutionalNetworkChecker.class);

    private INDArray generateOutput(String testFilePath, String modelFilePath) throws IOException, InterruptedException {
        var modelFile = new File(modelFilePath);
        var testFile = new File(testFilePath);
        var net = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        var reader = generateReader(testFile);

        ImagePreProcessingScaler normalizer = ModelSerializer.restoreNormalizerFromFile(modelFile);
        var batchSize = 1;
        var dataset = new RecordReaderDataSetIterator.Builder(reader, batchSize)
                .build();

        normalizer.fit(dataset);
        dataset.setPreProcessor(normalizer);
        return net.output(dataset);
    }


    private RecordReader generateReader(File file) throws IOException {
        var height = 30;
        var width = 30;
        var channels = 3;

        var reader = new ImageRecordReader(height,width,channels);
        var split = new FileSplit(file);

        reader.initialize(split);

        return reader;
    }

    public void checkNetwork() throws InterruptedException, IOException {

         var res = generateOutput("data/dataset/Beagle/beagle_7.jpg","cnn.zip");
         logger.info(res.toString());
    }
}
