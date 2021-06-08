package org.abondar.experimental.ml4j.nlp.command;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.abondar.experimental.ml4j.nlp.command.iterator.BasicLineIteratorCommand;
import org.bytedeco.javacpp.tools.Slf4jLogger;

public class NlpCommandSwitcher extends CommandSwitcher {

    private static final Slf4jLogger logger = new Slf4jLogger(NlpCommandSwitcher.class);

    @Override
    public void executeCommand(String cmd) {
        //try {
            switch (NlpCommands.valueOf(cmd)) {
                case BLIC:
                    var blic = new BasicLineIteratorCommand();
                    executor.executeCommand(blic);
                    break;

            }
      //  } catch (InterruptedException ex) {
      //      logger.error(ex.getMessage());
    //        System.exit(1);
     //   }
    }
}
