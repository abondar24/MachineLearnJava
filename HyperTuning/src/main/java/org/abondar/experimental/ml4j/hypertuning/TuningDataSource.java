package org.abondar.experimental.ml4j.hypertuning;

import org.abondar.experimental.ml4j.hypertuning.columns.Columns;
import org.abondar.experimental.ml4j.hypertuning.columns.Countries;
import org.abondar.experimental.ml4j.hypertuning.columns.Genders;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.File;
import java.io.IOException;
import java.util.Properties;

import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.Age;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.Balance;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.CreditScore;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.CustomerId;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.EstimatedSalary;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.Exited;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.Gender;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.Geography;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.HasCrCard;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.IsActiveMember;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.NumOfProducts;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.RowNumber;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.Surname;
import static org.abondar.experimental.ml4j.hypertuning.columns.Columns.Tenure;

public class TuningDataSource implements DataSource {

    private static final int LABEL_INDEX = 11;

    private static final int NUM_CLASSES = 1;

    private static final Logger LOGGER = LoggerFactory.getLogger(TuningDataSource.class);


    private int minibatchSize;

    public TuningDataSource(){}

    @Override
    public void configure(Properties properties) {
        this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize","16"));
    }

    @Override
    public Object trainData() {
        try {
            var iterator = new RecordReaderDataSetIterator(prepareData(),minibatchSize,LABEL_INDEX,NUM_CLASSES);
            return splitData(iterator).getTrainIterator();
        } catch (IOException | InterruptedException ex){
              LOGGER.error(ex.getMessage());
            System.exit(2);
        }

        return null;
    }

    @Override
    public Object testData() {
        try {
            var iterator = new RecordReaderDataSetIterator(prepareData(),minibatchSize,LABEL_INDEX,NUM_CLASSES);
            return splitData(iterator).getTestIterator();
        } catch (IOException | InterruptedException ex){
            LOGGER.error(ex.getMessage());
            System.exit(2);
        }

        return null;
    }

    @Override
    public Class<?> getDataType() {
        return DataSetIterator.class;
    }

    private DataSetIteratorSplitter splitData(DataSetIterator iterator) throws IOException,InterruptedException{
        var normalization = new NormalizerStandardize();
        normalization.fit(iterator);
        iterator.setPreProcessor(normalization);

        var splits = 1000;
        var ratio = 0.8;

        return new DataSetIteratorSplitter(iterator,splits,ratio);
    }

    private RecordReader prepareData() throws IOException, InterruptedException {
       var transform = buildTransform();

       var skipLines = 1;
       var delim = ',';
       var reader = new CSVRecordReader(skipLines,delim);

       var file =  new File("data/data.csv");
       var split = new FileSplit(file);
       reader.initialize(split);

       return new TransformProcessRecordReader(reader,transform);
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

    private TransformProcess buildTransform() {
        return  new TransformProcess.Builder(buildSchema())
                .removeColumns(RowNumber.name(), CustomerId.name(), Surname.name())
                .categoricalToInteger(Gender.name())
                .categoricalToOneHot(Geography.name())
                .removeColumns("Geography[France]")
                .build();
    }
}


