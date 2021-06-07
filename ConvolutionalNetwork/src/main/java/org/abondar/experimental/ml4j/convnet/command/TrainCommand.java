package org.abondar.experimental.ml4j.convnet.command;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.convnet.net.ConvolutionalNetwork;
import org.bytedeco.javacpp.tools.Slf4jLogger;

import java.io.IOException;

public class TrainCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(TrainCommand.class);

        try {
            var net = new ConvolutionalNetwork();
            net.buildModel("data/dataset.zip");

        } catch (IOException | InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }
    }
}
