package org.abondar.experimental.ml4j.nlp.net;

import java.util.Random;

public class NetConstants {
    public static final String DATA_PATH = "data/";
    private static final String DATASET_TAR_GZ = "aclImdb_v1.tar.gz";

    private static final String VECTOR_GZ = "GoogleNews-vectors-negative300.bin.gz";

    public static final String IMDB_TAR_GZ_PATH = DATA_PATH+DATASET_TAR_GZ;

    public static final String DATASET_URL = "http://ai.stanford.edu/~amaas/data/sentiment/" + DATASET_TAR_GZ;

    public static final String VECTOR_URL = "https://s3.amazonaws.com/dl4j-distribution/" + VECTOR_GZ;

    public static final String VECTOR_GZ_PATH = DATA_PATH + VECTOR_GZ;

    public static final String TRAIN_PATH = DATA_PATH + "aclImdb/train/";

    public static final String TEST_PATH = DATA_PATH + "aclImdb/test/";

    public static final int MAX_SENTENCE_LEN = 256;

    public static final Random RANGE = new Random(12345);



    private NetConstants(){}
}
