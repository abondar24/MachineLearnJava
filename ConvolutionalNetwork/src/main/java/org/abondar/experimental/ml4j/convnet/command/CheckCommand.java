package org.abondar.experimental.ml4j.convnet.command;


import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.convnet.net.ConvolutionalNetworkChecker;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class CheckCommand implements Command {
    @Override
    public void execute() {
        var logger = LoggerFactory.getLogger(CheckCommand.class);

        var nc = new ConvolutionalNetworkChecker();

        try {
            nc.checkNetwork();
        } catch (IOException | InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }
    }
}
