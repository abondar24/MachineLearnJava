package org.abondar.experimental.ml4j.data.command.reader;

import org.abondar.experimental.ml4j.command.Command;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class TransformReaderCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(TransformReaderCommand.class);
        var skipRows= 1;
        var delim = ',';

        var file = new File("data/csv/titanic.csv");
        var split = new FileSplit(file);

        var schema = new Schema.Builder()
                .addColumnInteger("Survived")
                .addColumnCategorical("Pclass", List.of("1","2","3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex",List.of("male","female"))
                .addColumnsInteger("Age","Siblings/Spouses Aboard","Parents/Children Aboard")
                .addColumnDouble("Fare")
                .build();

        var transform = new TransformProcess.Builder(schema)
                .removeColumns("Name","Fare")
                .categoricalToInteger("Sex")
                .categoricalToOneHot("Pclass")
                .removeColumns("Pclass[1]")
                .build();


        try {
            var reader = new CSVRecordReader(skipRows,delim);
            reader.initialize(split);

            var transformReader = new TransformProcessRecordReader(reader,transform);

            while (transformReader.hasNext()){
                transformReader.next().forEach(rec->
                        logger.info(rec.toString()));
            }

        } catch (IOException | InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }
    }
}
