package org.abondar.experimental.ml4j.nlp.command.word2vec;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.nlp.net.ImdbNet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class ImdbCommand implements Command {

    private static final Logger logger = LoggerFactory.getLogger(ImdbCommand.class);

    @Override
    public void execute() {
        var net = new ImdbNet();

        try {
          var model = net.buildModel();
          net.makeSentencePrediction("data/aclImdb/test/neg/0_2.txt",model);
        } catch (IOException ex) {
            logger.error(ex.getMessage());
            System.exit(2);
        }

    }
}
