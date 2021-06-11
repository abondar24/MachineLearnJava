package org.abondar.experimental.ml4j.nlp.command.iterator;

import org.abondar.experimental.ml4j.command.Command;


import org.deeplearning4j.nlp.uima.sentenceiterator.UimaSentenceIterator;

public class UimaIteratorCommand extends IteratorCommand implements Command {
    @Override
    public void execute() {

        try {
            var iterator = UimaSentenceIterator.createWithPath("data/files");
            printIterator(iterator);
        } catch (Exception ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }

    }
}
