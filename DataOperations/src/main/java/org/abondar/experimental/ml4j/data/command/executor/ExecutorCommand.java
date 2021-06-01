package org.abondar.experimental.ml4j.data.command.executor;

import org.abondar.experimental.ml4j.command.Command;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ExecutorCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(ExecutorCommand.class);

        var skipLines = 1;
        var csvDelim = ',';

        var file = new File("data/csv/titanic.csv");
        var split = new FileSplit(file);
        var reader = new CSVRecordReader(skipLines,csvDelim);

        try {
            reader.initialize(split);
        } catch (IOException| InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }

        var schema = new Schema.Builder()
                .addColumnInteger("Survived")
                .addColumnCategorical("Pclass", List.of("1","2","3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex",List.of("male","female"))
                .addColumnsInteger("Age","Siblings/Spouses Aboard","Parents/Children Aboard")
                .addColumnDouble("Fare")
                .build();

        var transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("Name","Fare")
                .categoricalToInteger("Sex")
                .categoricalToOneHot("Pclass")
                .removeColumns("Pclass[1]")
                .build();

        List<List<Writable>> output = new ArrayList<>();

        var writer = new CSVRecordWriter();
        var partitioner = new NumberOfRecordsPartitioner();
        var execFile = new File("data/csv/execute.csv");
        var execSplit = new FileSplit(execFile);

        try {
            writer.initialize(execSplit,partitioner);
        } catch (Exception ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }

        while (reader.hasNext()){
            output.add(reader.next());
        }

        var transformedOutput = LocalTransformExecutor.execute(output,transformProcess);

        try {
            writer.writeBatch(transformedOutput);
            writer.close();
        } catch (IOException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }

    }
}
