package org.abondar.experimental.ml4j.nlp.command.iterator;

import org.abondar.experimental.ml4j.command.Command;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;

import java.io.File;

public class FileIteratorCommand extends IteratorCommand implements Command {
    @Override
    public void execute() {
        var files = new File("data/files");
        var iterator = new FileSentenceIterator(files);

        printIterator(iterator);
    }
}
