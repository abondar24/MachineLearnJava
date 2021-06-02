package org.abondar.experimental.ml4j.deepnet.command;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.deepnet.net.DeepNetChecker;
import org.bytedeco.javacpp.tools.Slf4jLogger;

import java.io.IOException;

public class CheckCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(CheckCommand.class);

        var nc = new DeepNetChecker();

        try {
            nc.checkNetwork();
        } catch (IOException | InterruptedException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }
    }
}
