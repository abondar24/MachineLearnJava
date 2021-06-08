package org.abondar.experimental.ml4j.nlp.command.iterator;

import org.abondar.experimental.ml4j.nlp.preprocessor.SentenceDataPreProcessor;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

public class IteratorCommand {

    protected final Slf4jLogger logger = new Slf4jLogger(CollectionIteratorCommand.class);

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
