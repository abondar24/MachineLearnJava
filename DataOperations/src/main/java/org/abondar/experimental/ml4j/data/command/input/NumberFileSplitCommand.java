package org.abondar.experimental.ml4j.data.command.input;

import org.abondar.experimental.ml4j.command.Command;
import org.datavec.api.split.NumberedFileInputSplit;

import java.util.Arrays;

public class NumberFileSplitCommand implements Command {
    @Override
    public void execute() {
        var minId=1;
        var maxId=4;
        var numberSplit = new NumberedFileInputSplit("data/numbered/file%d.txt",
                minId,maxId);

        Arrays.stream(numberSplit.locations()).forEach(System.out::println);
    }
}
