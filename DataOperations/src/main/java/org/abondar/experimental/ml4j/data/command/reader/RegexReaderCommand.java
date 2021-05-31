package org.abondar.experimental.ml4j.data.command.reader;

import org.abondar.experimental.ml4j.command.Command;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.datavec.api.records.reader.impl.regex.RegexSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;

import java.io.IOException;

public class RegexReaderCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(RegexReaderCommand.class);
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
