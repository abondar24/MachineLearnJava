package org.abondar.experimental.ml4j.nlp.command.iterator;

import org.abondar.experimental.ml4j.command.Command;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CollectionIteratorCommand extends IteratorCommand implements Command {
    @Override
    public void execute() {

        List<String> sentences = new ArrayList<>();
        try {
            sentences = readFile("data/raw_sentences.txt");
        } catch (IOException ex) {
            logger.error(ex.getMessage());
            System.exit(2);
        }

        if (sentences.isEmpty()) {
            logger.error("Empty file contents");
            System.exit(1);
        }

        var iterator = new CollectionSentenceIterator(sentences);
        printIterator(iterator);

    }

    private List<String> readFile(String filename) throws IOException {
        List<String> sentences = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            while (br.ready()) {
                sentences.add(br.readLine());
            }
        }

        return sentences;

    }
}
