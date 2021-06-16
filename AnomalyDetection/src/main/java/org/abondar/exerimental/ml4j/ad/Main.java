package org.abondar.exerimental.ml4j.ad;

import org.abondar.exerimental.ml4j.ad.mnist.MnistNet;

public class Main {

    public static void main(String[] args) {
        var mnist = new MnistNet();
        mnist.runComputations();
    }
}

