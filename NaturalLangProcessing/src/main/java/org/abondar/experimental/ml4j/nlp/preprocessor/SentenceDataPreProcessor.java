package org.abondar.experimental.ml4j.nlp.preprocessor;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

import java.util.Locale;

public class SentenceDataPreProcessor {
    public static void setPreProcessor(SentenceIterator iterator){
        iterator.setPreProcessor(sentence-> sentence.toLowerCase(Locale.ROOT));
    }
}
