package org.abondar.experimental.ml4j.nlp.command.word2vec;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.nlp.preprocessor.SentenceDataPreProcessor;



import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

//for new apis
//import org.nd4j.common.primitives.Pair;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class Word2VecCommand implements Command {

    private static final Logger logger = LoggerFactory.getLogger(Word2VecCommand.class);


    private SentenceIterator getIterator(String filename){
        var file = new File(filename);
        var iterator = new LineSentenceIterator(file);
        SentenceDataPreProcessor.setPreProcessor(iterator);

        return iterator;
    }

    private TokenizerFactory getTokeinzer(){
        var tokenizer = new DefaultTokenizerFactory();
        tokenizer.setTokenPreProcessor(new EndingPreProcessor());
        return tokenizer;
    }

    private Word2Vec trainModel(SentenceIterator iterator,TokenizerFactory tokenizer){

        var model = new Word2Vec.Builder()
                .iterate(iterator)
                .tokenizerFactory(tokenizer)
                .minWordFrequency(5)
                .layerSize(100)
                .seed(42)
                .epochs(50)
                .windowSize(5)
                .build();

        logger.info("Training Word2Vec Model");
        model.fit();

        return model;
    }

    private void evaluateModel(Word2Vec model){
        var words = model.wordsNearest("money",10);
        words.forEach(logger::info);

        var similarity = model.similarity("money","people");
        var msg = String.format("Cos Similarity for words 'money' and 'people': %f",similarity);
        logger.info(msg);
    }

    private void saveModel(Word2Vec model,File vectorFile) throws IOException {
        WordVectorSerializer.writeWordVectors(model.lookupTable(),vectorFile);
        WordVectorSerializer.writeWord2VecModel(model,"model.zip");
    }

    private void visualizeData(File vectorFile) throws IOException{
        List<String> uniqueWords = new ArrayList<>();
        var outputFilename = "tsne-coords.csv";

        Pair<InMemoryLookupTable, VocabCache> vectors = WordVectorSerializer.loadTxt(vectorFile);
        VocabCache cache = vectors.getSecond();
        INDArray weights = vectors.getFirst().getSyn0();

        for (int i=0;i<cache.numWords();i++){
            uniqueWords.add(cache.wordAtIndex(i));
        }

        var tsne = buildTsne();
        tsne.fit(weights);
        tsne.saveAsFile(uniqueWords,outputFilename);

    }

    private BarnesHutTsne buildTsne(){
      return new BarnesHutTsne.Builder()
              .setMaxIter(100)
              .theta(0.5)
              .normalize(false)
              .learningRate(500)
              .useAdaGrad(false)
              .build();
    }

    @Override
    public void execute() {
        var filename = "data/raw_sentences_large.txt";

        var iterator = getIterator(filename);
        var tokenizer = getTokeinzer();
        var model = trainModel(iterator,tokenizer);

        evaluateModel(model);

        var vectorFile = new File("words.txt");
        try{
            saveModel(model,vectorFile);
            visualizeData(vectorFile);
        } catch (IOException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }

    }
}
