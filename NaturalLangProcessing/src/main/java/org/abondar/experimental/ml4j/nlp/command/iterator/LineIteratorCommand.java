package org.abondar.experimental.ml4j.nlp.command.iterator;

import org.abondar.experimental.ml4j.command.Command;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;

import java.io.File;

public class LineIteratorCommand extends IteratorCommand implements Command {
    @Override
    public void execute() {
        var file = new File("data/raw_sentences.txt");
        var iterator = new LineSentenceIterator(file);

        printIterator(iterator);
    }
}
