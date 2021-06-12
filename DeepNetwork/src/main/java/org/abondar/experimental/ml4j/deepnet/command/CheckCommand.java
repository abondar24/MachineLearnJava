package org.abondar.experimental.ml4j.deepnet.command;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.deepnet.net.DeepNetChecker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class CheckCommand implements Command {

    private static final Logger logger = LoggerFactory.getLogger(CheckCommand.class);

    @Override
    public void execute() {
        var nc = new DeepNetChecker();

        try {
            nc.checkNetwork();
        } catch (IOException | InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }
    }
}
