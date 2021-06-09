package org.abondar.experimental.ml4j.nlp.command;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.abondar.experimental.ml4j.nlp.command.iterator.BasicLineIteratorCommand;
import org.abondar.experimental.ml4j.nlp.command.iterator.CollectionIteratorCommand;
import org.abondar.experimental.ml4j.nlp.command.iterator.FileIteratorCommand;
import org.abondar.experimental.ml4j.nlp.command.iterator.LineIteratorCommand;
import org.abondar.experimental.ml4j.nlp.command.iterator.UimaIteratorCommand;
import org.abondar.experimental.ml4j.nlp.command.word2vec.Word2VecCommand;
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

                case CIC:
                    var cic = new CollectionIteratorCommand();
                    executor.executeCommand(cic);
                    break;

                case FIC:
                    var fic = new FileIteratorCommand();
                    executor.executeCommand(fic);
                    break;

                case LIC:
                    var lic = new LineIteratorCommand();
                    executor.executeCommand(lic);
                    break;

                case UIC:
                    var uic = new UimaIteratorCommand();
                    executor.executeCommand(uic);
                    break;

                case W2VC:
                    var w2vc = new Word2VecCommand();
                    executor.executeCommand(w2vc);
                    break;

            }
        } catch (IllegalArgumentException ex) {
            logger.error(ex.getMessage());
            System.exit(1);
        }
    }
}
