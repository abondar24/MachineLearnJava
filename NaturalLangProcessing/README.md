# Natural Language Processing

Examples of nlp

## Examples

1. Basic Line Iterator (blic) - Simple single line sentence iterator.
2. Line Iterator (lic) - Multi-Sentence iterator.
3. Collection Iterator (cic) - Iterator working with list of strings.
4. File Iterator (fic) - Iterator working with directories and files.
5. UIMA Iterator (uic) - UIMA iterator working with directories and segmenting the sentences.
6. Word2Vec (w2vc) - Word2Vec training example
7. IMDB (imdb) - Sentence classification using Word2Vec and Convolutional Neural Network.

## Datasets

1. [IMDB review data](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
2. [Google News Vectors](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz)


## Build And Run
```
mvn clean install

java -jar nlp.jar <command>
```

# Note

If RAM size on the pc is 8GB use vm parameters from below

```
-Xmx2G -Xmx6G
```

Command names are in ().
