package org.abondar.experimental.ml4j.data.command.normalization;

import org.abondar.experimental.ml4j.command.Command;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class NormalizationCommand implements Command {

    private static final Logger logger = LoggerFactory.getLogger(NormalizationCommand.class);

    @Override
    public void execute() {

        var batchSize = 2;
        var skipRows = 1;
        var delim = ',';

        var file = new File("data/csv/titanic.csv");
        var split = new FileSplit(file);

        var schema = new Schema.Builder()
                .addColumnInteger("Survived")
                .addColumnCategorical("Pclass", List.of("1", "2", "3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex", List.of("male", "female"))
                .addColumnsInteger("Age", "Siblings/Spouses Aboard", "Parents/Children Aboard")
                .addColumnDouble("Fare")
                .build();

        var transform = new TransformProcess.Builder(schema)
                .removeColumns("Name", "Fare")
                .categoricalToInteger("Sex")
                .categoricalToOneHot("Pclass")
                .removeColumns("Pclass[1]")
                .build();

        var reader = new CSVRecordReader(skipRows, delim);

        try {
            reader.initialize(split);
        } catch (IOException | InterruptedException ex) {
            logger.error(ex.getMessage());
            System.exit(2);
        }

        var transformReader = new TransformProcessRecordReader(reader, transform);
        var normalization = new NormalizerStandardize();
        var iterator = new RecordReaderDataSetIterator(transformReader, batchSize);

        logger.info("Before normalization");
        logger.info(iterator.next().getFeatures().toString());

        iterator.reset();
        normalization.fit(iterator);
        iterator.setPreProcessor(normalization);

        logger.info("After normalization");
        logger.info(iterator.next().getFeatures().toString());

    }

}
