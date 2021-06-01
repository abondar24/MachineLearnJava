package org.abondar.experimental.ml4j.deepnet;

import org.bytedeco.javacpp.tools.Slf4jLogger;

import java.io.IOException;

public class Main {

    private static final Slf4jLogger logger = new Slf4jLogger(Main.class);

    public static void main(String[] args) {

       try {
           var net = new MultilayerNetwork();
           net.buildModel("/data/model.csv");

       } catch (IOException | InterruptedException ex){
           logger.error(ex.getMessage());
           System.exit(2);
       }

    }
}
