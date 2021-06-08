package org.abondar.experimental.ml4j.nlp.command.iterator;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.nlp.preprocessor.SentenceDataPreProcessor;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;

import java.io.IOException;

public class BasicLineIteratorCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(BasicLineIteratorCommand.class);
        int count = 0;

        try {
            var iterator = new BasicLineIterator("data/raw_sentences.txt");

            while (iterator.hasNext()) {
                iterator.nextSentence();
                count++;
            }
            var msg = String.format("Count - %d",count);
            logger.info(msg);

            iterator.reset();
            SentenceDataPreProcessor.setPreProcessor(iterator);
        } catch (IOException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }

    }
}
