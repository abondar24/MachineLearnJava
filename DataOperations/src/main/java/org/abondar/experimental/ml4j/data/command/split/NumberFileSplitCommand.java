package org.abondar.experimental.ml4j.data.command.split;

import org.abondar.experimental.ml4j.command.Command;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.datavec.api.split.NumberedFileInputSplit;

import java.util.Arrays;

public class NumberFileSplitCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(NumberFileSplitCommand.class);
        var minId=1;
        var maxId=4;
        var numberSplit = new NumberedFileInputSplit("data/numbered/file%d.txt",
                minId,maxId);

        Arrays.stream(numberSplit.locations()).forEach(ur-> logger.info(ur.toString()));
    }
}
