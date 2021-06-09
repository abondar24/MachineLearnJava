package org.abondar.experimental.ml4j.nlp.command.iterator;

import org.abondar.experimental.ml4j.nlp.preprocessor.SentenceDataPreProcessor;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IteratorCommand {

    protected final Logger logger = LoggerFactory.getLogger(IteratorCommand.class);

    protected void printIterator(SentenceIterator iterator){
        int count = 0;
        while (iterator.hasNext()) {
            iterator.nextSentence();
            count++;
        }
        var msg = String.format("Count - %d", count);
        logger.info(msg);

        iterator.reset();
        SentenceDataPreProcessor.setPreProcessor(iterator);
        while (iterator.hasNext()) {
            logger.info(iterator.nextSentence());
        }
    }
}
