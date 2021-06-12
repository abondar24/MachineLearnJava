package org.abondar.experimental.ml4j.convnet.command;


import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NetCommandSwitcher extends CommandSwitcher {
    private static final Logger LOGGER = LoggerFactory.getLogger(NetCommandSwitcher.class);
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
            LOGGER.error(ex.getMessage());
            System.exit(1);
        }
    }
}
