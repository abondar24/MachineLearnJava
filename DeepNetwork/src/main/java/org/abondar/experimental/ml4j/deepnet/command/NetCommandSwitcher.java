package org.abondar.experimental.ml4j.deepnet.command;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.bytedeco.javacpp.tools.Slf4jLogger;

public class NetCommandSwitcher extends CommandSwitcher {
    private static final Slf4jLogger LOGGER = new Slf4jLogger(NetCommandSwitcher.class);
    @Override
    public void executeCommand(String cmd) {
        try {

            switch (NetCommands.valueOf(cmd)) {
                case TRAIN:
                     var trc = new TrainCommand();
                     executor.executeCommand(trc);
                     break;

                case CHECK:
                    var cc = new CheckCommand();
                    executor.executeCommand(cc);
            }

        } catch (IllegalArgumentException ex) {
            LOGGER.error("Check documentation for command list");
            System.exit(1);
        }
    }
}
