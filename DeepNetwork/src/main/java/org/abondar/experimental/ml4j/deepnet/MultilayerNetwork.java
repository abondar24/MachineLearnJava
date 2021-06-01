package org.abondar.experimental.ml4j.deepnet;

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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;

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

public class MultilayerNetwork {

    private static final Slf4jLogger LOGGER = new Slf4jLogger(MultilayerNetwork.class);

    private RecordReader createReader(String filepath) throws IOException, InterruptedException {
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

    private DataSetIteratorSplitter getDataset(RecordReader reader) {
        var batchSize = 1;
        var labelIndex = 1;
        var numClasses = 2;

        var datasetIterator = new RecordReaderDataSetIterator.Builder(reader, batchSize)
                .classification(labelIndex, numClasses)
                .build();

        return normalizeData(datasetIterator) ;
    }

    private DataSetIteratorSplitter normalizeData(DataSetIterator dataSetIterator) {
        var normalizer = new NormalizerStandardize();
        normalizer.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(normalizer);

        var totalBatches = 1250;
        var ratio = 0.8;
        return new DataSetIteratorSplitter(dataSetIterator, totalBatches, ratio);
    }

    public void buildModel(String filePath) throws IOException, InterruptedException {
        var reader = createReader(filePath);
        var schema = buildSchema();

        var transformReader = applyTransform(schema, reader);
        var dataset = getDataset(transformReader);
    }



}
