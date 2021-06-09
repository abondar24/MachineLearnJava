package org.abondar.experimental.ml4j.deepnet.command;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.deepnet.net.DeepNetwork;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class TrainCommand implements Command {
    @Override
    public void execute() {
        var logger = LoggerFactory.getLogger(TrainCommand.class);

        try {
            var net = new DeepNetwork();
            net.buildModel("data/model.csv");

        } catch (IOException | InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }
    }
}
