package org.abondar.experimental.ml4j.convnet.command;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.convnet.net.ConvolutionalNetwork;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class TrainCommand implements Command {
    @Override
    public void execute() {
        var logger = LoggerFactory.getLogger(TrainCommand.class);

        try {
            var net = new ConvolutionalNetwork();
            net.buildModel("data/dataset.zip");

        } catch (IOException | InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }
    }
}
