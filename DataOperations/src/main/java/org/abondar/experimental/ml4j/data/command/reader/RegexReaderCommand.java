package org.abondar.experimental.ml4j.data.command.reader;

import org.abondar.experimental.ml4j.command.Command;
import org.datavec.api.records.reader.impl.regex.RegexSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class RegexReaderCommand implements Command {

    private static final Logger logger = LoggerFactory.getLogger(RegexReaderCommand.class);

    @Override
    public void execute() {
        var split = new NumberedFileInputSplit("data/log/log-%d.log", 1, 1);
        var regex = "(\\d{2}/\\d{2}/\\d{2}) (\\d{2}:\\d{2}:\\d{2}) ([A-Z]) (.*)";

        var sequenceReader = new RegexSequenceRecordReader(regex, 0);

        try {
            sequenceReader.initialize(split);
            while (sequenceReader.hasNext()){
                sequenceReader.next().forEach(r->{
                    logger.info(r.toString());
                });
            }

        } catch (IOException | InterruptedException ex) {
            logger.error(ex.getMessage());
            System.exit(2);

        }


    }
}
