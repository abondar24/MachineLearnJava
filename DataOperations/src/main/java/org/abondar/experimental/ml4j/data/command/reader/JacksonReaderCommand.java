package org.abondar.experimental.ml4j.data.command.reader;

import org.abondar.experimental.ml4j.command.Command;
import org.datavec.api.records.reader.impl.jackson.FieldSelection;
import org.datavec.api.records.reader.impl.jackson.JacksonLineRecordReader;
import org.datavec.api.split.FileSplit;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;


public class JacksonReaderCommand implements Command {
    private static final Logger logger = LoggerFactory.getLogger(JacksonReaderCommand.class);

    @Override
    public void execute() {

        var fields = new FieldSelection.Builder()
                .addField("delta")
                .addField("time")
                .addField("number")
                .build();

        var mapper = new ObjectMapper(new JsonFactory());
        mapper.configure(JsonParser.Feature.AUTO_CLOSE_SOURCE, true);

        var reader = new JacksonLineRecordReader(fields, mapper);


        try {
            var file = new File("data/json/test_json.txt");
            reader.initialize(new FileSplit(file));

            while (reader.hasNext()) {
                reader.next().forEach(r -> logger.info(r.toString()));
            }


        } catch (IOException | InterruptedException ex) {
            logger.error(ex.getMessage());
            System.exit(2);

        }
    }
}
