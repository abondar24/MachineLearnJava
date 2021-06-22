package org.abondar.experimental.ml4j.hypertuning;

import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.arbiter.ui.listener.ArbiterStatusListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

public class TuningArbiter {

    private static final Logger LOGGER = LoggerFactory.getLogger(TuningArbiter.class);

    public void performTuning() {
        var conf = buildOptimizationConfig();

        var taskCreator = new MultiLayerNetworkTaskCreator();
        var runner = new LocalOptimizationRunner(conf, taskCreator);

        storeData(runner);
        runner.execute();
        printScores(runner);
    }

    private void storeData(IOptimizationRunner runner) {
        var storageFile = new File("data/tuning.dl4j");
        var storage = new FileStatsStorage(storageFile);

        var listener = new ArbiterStatusListener(storage);
        runner.addListeners(listener);

        UIServer.getInstance().attach(storage);
    }

    private void printScores(IOptimizationRunner runner) {
        var bestScore = runner.bestScore();
        int bestCandidateIndex = runner.bestScoreCandidateIndex();
        int numberOfConfigs = runner.numCandidatesCompleted();

        LOGGER.info(String.format("Best score %f", bestScore));
        LOGGER.info(String.format("Index of model with best score %d",bestCandidateIndex));
        LOGGER.info(String.format("Number of configurations evaluated %d",numberOfConfigs));
    }


    private OptimizationConfiguration buildOptimizationConfig() {
        return new OptimizationConfiguration.Builder()
                .candidateGenerator(getCandidateGenerator())
                .dataSource(TuningDataSource.class, getDataSourceProperties())
                .modelSaver(getModelSaver())
                .scoreFunction(getScoreFunction())
                .terminationConditions(getTermConditions())
                .build();
    }

    private CandidateGenerator getCandidateGenerator() {
        return new RandomSearchGenerator(configureHyperSpace(), getDataParams());
    }

    private MultiLayerSpace configureHyperSpace() {
        var layerSize = getLayerSize();

        return new MultiLayerSpace.Builder()
                .updater(new AdamSpace(getLearningRate()))
                .addLayer(buildDenseLayerSpace(11, layerSize))
                .addLayer(buildDenseLayerSpace(layerSize, layerSize))
                .addLayer(buildOutputLayerSpace())
                .build();
    }


    private ParameterSpace<Double> getLearningRate() {
        var learnMin = 0.0001;
        var learnMax = 0.1;
        return new ContinuousParameterSpace(learnMin, learnMax);
    }

    private ParameterSpace<Integer> getLayerSize() {
        var layerMin = 5;
        var layerMax = 11;
        return new IntegerParameterSpace(5, 11);
    }


    private DenseLayerSpace buildDenseLayerSpace(int nIn, ParameterSpace<Integer> nOut) {
        return new DenseLayerSpace.Builder()
                .activation(Activation.RELU)
                .nIn(nIn)
                .nOut(nOut)
                .build();
    }

    private DenseLayerSpace buildDenseLayerSpace(ParameterSpace<Integer> nIn, ParameterSpace<Integer> nOut) {
        return new DenseLayerSpace.Builder()
                .activation(Activation.RELU)
                .nIn(nIn)
                .nOut(nOut)
                .build();
    }

    private OutputLayerSpace buildOutputLayerSpace() {
        return new OutputLayerSpace.Builder()
                .activation(Activation.SIGMOID)
                .lossFunction(LossFunctions.LossFunction.XENT)
                .nOut(1)
                .build();
    }

    private Map<String, Object> getDataParams() {
        Map<String, Object> dataParams = new HashMap<>();
        dataParams.put("batchSize", 10);

        return dataParams;
    }

    private Properties getDataSourceProperties() {
        var props = new Properties();
        props.setProperty("minibatchSize", "64");
        return props;
    }

    private ResultSaver getModelSaver() {
        return new FileModelSaver("data/");
    }

    private ScoreFunction getScoreFunction() {
        return new EvaluationScoreFunction(Evaluation.Metric.ACCURACY);
    }

    private TerminationCondition[] getTermConditions() {
        return new TerminationCondition[]{
                new MaxTimeCondition(120, TimeUnit.MINUTES),
                new MaxCandidatesCondition(30)
        };
    }
}
