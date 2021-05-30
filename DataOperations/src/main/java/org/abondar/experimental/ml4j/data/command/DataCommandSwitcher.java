package org.abondar.experimental.ml4j.data.command;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.abondar.experimental.ml4j.data.command.split.FileSplitCommand;
import org.abondar.experimental.ml4j.data.command.split.NumberFileSplitCommand;
import org.abondar.experimental.ml4j.data.command.split.TransformSplitCommand;
import org.bytedeco.javacpp.tools.Slf4jLogger;

public class DataCommandSwitcher extends CommandSwitcher {

    private static final Slf4jLogger logger = new Slf4jLogger(DataCommandSwitcher.class);
    @Override
    public void executeCommand(String cmd) {

        try {
            switch (DataCommands.valueOf(cmd)){
                case FSC:
                    FileSplitCommand fsc = new FileSplitCommand();
                    executor.executeCommand(fsc);
                    break;

                case NFSC:
                    NumberFileSplitCommand nfsc = new NumberFileSplitCommand();
                    executor.executeCommand(nfsc);
                    break;

                case TSC:
                    TransformSplitCommand tsc = new TransformSplitCommand();
                    executor.executeCommand(tsc);
                    break;
            }
        } catch (IllegalArgumentException ex) {
           logger.error("Check documentation for command list");
            System.exit(1);
        }
    }
}
