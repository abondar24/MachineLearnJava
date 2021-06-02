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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;

import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Age;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Balance;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.CreditScore;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.CustomerId;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.EstimatedSalary;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Gender;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Geography;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.HasCrCard;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.IsActiveMember;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.NumOfProducts;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.RowNumber;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Surname;
import static org.abondar.experimental.ml4j.deepnet.columns.Columns.Tenure;

public class DeepNetChecker {
    private static final Slf4jLogger LOGGER = new Slf4jLogger(DeepNetChecker.class);

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
                .build();
    }

    private RecordReader initReader(String filepath) throws IOException, InterruptedException {
        var csvSkipLine = 1;
        var csvDelim = ',';

        var reader = new CSVRecordReader(csvSkipLine, csvDelim);
        var file = new File(filepath);
        var split = new FileSplit(file);

        reader.initialize(split);

        return applyTransform(buildSchema(), reader);
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

    private INDArray generateOutput(String testFilePath, String modelFilePath) throws IOException, InterruptedException {
        var modelFile = new File(modelFilePath);
        var net = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        var reader = initReader(testFilePath);

        NormalizerStandardize normalizer = ModelSerializer.restoreNormalizerFromFile(modelFile);
        var batchSize = 1;
        var dataset = new RecordReaderDataSetIterator.Builder(reader, batchSize)
                .build();

        normalizer.fit(dataset);
        dataset.setPreProcessor(normalizer);
        return net.output(dataset);
    }

    public void checkNetwork() throws IOException, InterruptedException {
        INDArray indArray = generateOutput("data/test.csv","model.zip");

        var msg = "";
        for (int i=0;i<indArray.rows();i++){
            if (indArray.getDouble(i,0)>indArray.getDouble(i,1)){
                msg = String.format("Customer %d is Happy",i+1);
            } else {
                msg = String.format("Customer %d is not Happy",i+1);
            }
            LOGGER.info(msg);
        }
    }
}
