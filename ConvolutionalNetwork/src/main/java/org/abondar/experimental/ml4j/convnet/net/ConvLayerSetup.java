package org.abondar.experimental.ml4j.convnet.net;

public class ConvLayerSetup {

    private final int layerNum;
    private final int kernelSize;
    private final int outNodes;

    public ConvLayerSetup(int layerNum, int kernelSize, int outNodes) {
        this.layerNum = layerNum;
        this.kernelSize = kernelSize;
        this.outNodes = outNodes;
    }

    public int getLayerNum() {
        return layerNum;
    }

    public int getKernelSize() {
        return kernelSize;
    }

    public int getOutNodes() {
        return outNodes;
    }
}
