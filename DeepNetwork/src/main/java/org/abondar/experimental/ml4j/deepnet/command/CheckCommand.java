package org.abondar.experimental.ml4j.deepnet.command;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.deepnet.net.DeepNetChecker;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class CheckCommand implements Command {
    @Override
    public void execute() {
        var logger = LoggerFactory.getLogger(CheckCommand.class);

        var nc = new DeepNetChecker();

        try {
            nc.checkNetwork();
        } catch (IOException | InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }
    }
}
