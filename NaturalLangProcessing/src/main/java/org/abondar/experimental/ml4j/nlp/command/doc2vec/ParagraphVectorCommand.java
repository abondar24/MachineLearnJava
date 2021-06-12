package org.abondar.experimental.ml4j.nlp.command.doc2vec;

import org.abondar.experimental.ml4j.command.Command;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CompositePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class ParagraphVectorCommand implements Command {

    private static final Logger logger = LoggerFactory.getLogger(ParagraphVectorCommand.class);


    @Override
    public void execute() {
        var labeledIterator = buildIterator("data/labeled");
        var unlabeledIterator = buildIterator("data/unlabeled");

        var tokenizer = createTokenizer();
        var vectors = trainModel(labeledIterator, tokenizer);
        var lookUpTable = createLookupTable(vectors);

        predictLabels(labeledIterator,unlabeledIterator,lookUpTable,tokenizer);
    }

    private LabelAwareIterator buildIterator(String path) {
        return new FileLabelAwareIterator.Builder()
                .addSourceFolder(new File(path))
                .build();
    }


    private TokenizerFactory createTokenizer() {
        var tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        return tokenizerFactory;
    }

    private ParagraphVectors trainModel(LabelAwareIterator iterator, TokenizerFactory tokenizer) {
        var vectors = new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.005)
                .batchSize(1000)
                .epochs(5)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizer)
                .build();

        vectors.fit();

        return vectors;
    }


    private InMemoryLookupTable<VocabWord> createLookupTable(ParagraphVectors vectors) {
        return (InMemoryLookupTable<VocabWord>) vectors.getLookupTable();
    }

    private void predictLabels(LabelAwareIterator labeledIterator, LabelAwareIterator unlabeledIterator,
                               InMemoryLookupTable<VocabWord> lookupTable, TokenizerFactory tokenizer) {

        while (unlabeledIterator.hasNextDocument()) {
            var tokenizedDocument = getTokenizedDocument(unlabeledIterator, tokenizer);
            var vocabCache = lookupTable.getVocabCache();
            var count = countMatchedInstances(tokenizedDocument, vocabCache);
            var documentVector = storeWordVectors(count, vocabCache, lookupTable, tokenizedDocument);

            checkCosineSimilarity(labeledIterator,lookupTable,documentVector);

        }


    }

    private List<String> getTokenizedDocument(LabelAwareIterator iterator, TokenizerFactory tokenizer) {
        var labelledDoc = iterator.nextDocument();
        return tokenizer.create(labelledDoc.getContent()).getTokens();
    }

    private AtomicInteger countMatchedInstances(List<String> document, VocabCache<VocabWord> vocabCache) {
        AtomicInteger count = new AtomicInteger(0);

        document.stream().filter(vocabCache::containsWord)
                .forEach(w -> count.incrementAndGet());

        return count;
    }


    private INDArray storeWordVectors(AtomicInteger count, VocabCache<VocabWord> vocabCache,
                                      InMemoryLookupTable<VocabWord> lookupTable, List<String> document) {
        var allWords = Nd4j.create(count.get(), lookupTable.layerSize());
        count.set(0);

        document.stream().filter(vocabCache::containsWord)
                .forEach(word -> allWords.putRow(count.getAndIncrement(), lookupTable.vector(word)));

        return allWords.mean(0);
    }


    private void checkCosineSimilarity(LabelAwareIterator labeledIterator,InMemoryLookupTable<VocabWord> lookupTable,
                                       INDArray docVec){
        var labels = labeledIterator.getLabelsSource().getLabels();
        List<Pair<String, Double>> res = new ArrayList<>();

        labels.forEach(label->{
            var vecLabel = lookupTable.vector(label);
            if (vecLabel ==null){
                logger.error(String.format("Label '%s' has no known vector!",label));
                System.exit(2);
            }

            var sim = Transforms.cosineSim(docVec,vecLabel);
            res.add(new Pair<>(label, sim));
        });

        res.forEach(r-> logger.info(String.format("Prediction score %s:%f ",r.getFirst(),r.getSecond())));
    }

}
