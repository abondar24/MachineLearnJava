package org.abondar.experimental.ml4j.data.command.reader;

import org.abondar.experimental.ml4j.command.Command;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class CsvReaderCommand implements Command {

    private static final Logger logger = LoggerFactory.getLogger(CsvReaderCommand.class);

    @Override
    public void execute() {
        var skipRows= 1;
        var delim = ',';

        var file = new File("data/csv/titanic.csv");
        var split = new FileSplit(file);

        try {
            var reader = new CSVRecordReader(skipRows,delim);
            reader.initialize(split);

            while (reader.hasNext()){
                reader.nextRecord()
                        .getRecord()
                        .forEach(writable -> logger.info(writable.toString()));
            }

        } catch (IOException | InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }

    }
}
