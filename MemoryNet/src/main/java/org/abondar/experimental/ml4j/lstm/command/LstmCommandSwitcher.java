package org.abondar.experimental.ml4j.lstm.command;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LstmCommandSwitcher extends CommandSwitcher {

    private static final Logger logger = LoggerFactory.getLogger(LstmCommandSwitcher.class);

    @Override
    public void executeCommand(String cmd) {
        try {
            switch (LstmCommands.valueOf(cmd)) {
                case TSC:
                    var tsc = new TimeSeriesCommand();
                    executor.executeCommand(tsc);
                    break;

                case SC:
                    var sc = new SequenceDataCommand();
                    executor.executeCommand(sc);
                    break;

            }
        } catch (IllegalArgumentException ex) {
            logger.error(ex.getMessage());
            System.exit(1);
        }
    }
}
