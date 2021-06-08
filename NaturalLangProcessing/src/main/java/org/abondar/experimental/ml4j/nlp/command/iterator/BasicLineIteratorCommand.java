package org.abondar.experimental.ml4j.nlp.command.iterator;

import org.abondar.experimental.ml4j.command.Command;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;

import java.io.IOException;

public class BasicLineIteratorCommand extends IteratorCommand implements Command {
    @Override
    public void execute() {
        try {
            var iterator = new BasicLineIterator("data/raw_sentences.txt");

            printIterator(iterator);
        } catch (IOException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }

    }
}
