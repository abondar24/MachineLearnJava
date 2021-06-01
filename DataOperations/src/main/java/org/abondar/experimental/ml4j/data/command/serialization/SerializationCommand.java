package org.abondar.experimental.ml4j.data.command.serialization;

import org.abondar.experimental.ml4j.command.Command;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.string.ConvertToString;

import java.util.List;

public class SerializationCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(SerializationCommand.class);

        var schema = new Schema.Builder()
                .addColumnInteger("Survived")
                .addColumnCategorical("Pclass", List.of("1","2","3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex",List.of("male","female"))
                .addColumnsInteger("Age","Siblings/Spouses Aboard","Parents/Children Aboard")
                .addColumnDouble("Fare")
                .build();

        var transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("Fare")
                .transform(new ConvertToString("Survived"))
                .categoricalToInteger("Sex")
                .build();

        var json = transformProcess.toJson();
        logger.info(json);

        var yaml = transformProcess.toYaml();
        logger.info(yaml);
    }
}
