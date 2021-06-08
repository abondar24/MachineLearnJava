package org.abondar.experimental.ml4j.nlp.command.iterator;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.nlp.preprocessor.SentenceDataPreProcessor;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;

import java.io.File;

public class LineIteratorCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(LineIteratorCommand.class);
        int count = 0;

        var file = new File("data/raw_sentences.txt");
        var iterator = new LineSentenceIterator(file);

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
