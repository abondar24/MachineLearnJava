package org.abondar.experimental.ml4j.nlp.command;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.abondar.experimental.ml4j.nlp.command.iterator.BasicLineIteratorCommand;
import org.abondar.experimental.ml4j.nlp.command.iterator.LineIteratorCommand;
import org.bytedeco.javacpp.tools.Slf4jLogger;

public class NlpCommandSwitcher extends CommandSwitcher {

    private static final Slf4jLogger logger = new Slf4jLogger(NlpCommandSwitcher.class);

    @Override
    public void executeCommand(String cmd) {
        try {
            switch (NlpCommands.valueOf(cmd)) {
                case BLIC:
                    var blic = new BasicLineIteratorCommand();
                    executor.executeCommand(blic);
                    break;

                case LIC:
                    var lic = new LineIteratorCommand();
                    executor.executeCommand(lic);
                    break;

            }
        } catch (IllegalArgumentException ex) {
            logger.error(ex.getMessage());
            System.exit(1);
        }
    }
}
