package org.abondar.experimental.ml4j.data.command;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.abondar.experimental.ml4j.data.command.reader.CsvReaderCommand;
import org.abondar.experimental.ml4j.data.command.reader.ImageReaderCommand;
import org.abondar.experimental.ml4j.data.command.reader.JacksonReaderCommand;
import org.abondar.experimental.ml4j.data.command.reader.RegexReaderCommand;
import org.abondar.experimental.ml4j.data.command.reader.TransformReaderCommand;
import org.abondar.experimental.ml4j.data.command.serialization.SerializationCommand;
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
                case CSVRC:
                    var csvrc = new CsvReaderCommand();
                    executor.executeCommand(csvrc);
                    break;

                case FSC:
                    var fsc = new FileSplitCommand();
                    executor.executeCommand(fsc);
                    break;

                case IRC:
                    var irc = new ImageReaderCommand();
                    executor.executeCommand(irc);
                    break;

                case JRC:
                    var jrc = new JacksonReaderCommand();
                    executor.executeCommand(jrc);
                    break;

                case NFSC:
                    var nfsc = new NumberFileSplitCommand();
                    executor.executeCommand(nfsc);
                    break;

                case RRC:
                    var rrc = new RegexReaderCommand();
                    executor.executeCommand(rrc);
                    break;

                case SC:
                    var sc = new SerializationCommand();
                    executor.executeCommand(sc);
                    break;

                case TRC:
                    var trc = new TransformReaderCommand();
                    executor.executeCommand(trc);
                    break;

                case TSC:
                    var tsc = new TransformSplitCommand();
                    executor.executeCommand(tsc);
                    break;
            }
        } catch (IllegalArgumentException ex) {
           logger.error("Check documentation for command list");
            System.exit(1);
        }
    }
}
