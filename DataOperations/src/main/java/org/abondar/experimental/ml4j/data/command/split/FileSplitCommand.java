package org.abondar.experimental.ml4j.data.command.split;

import org.abondar.experimental.ml4j.command.Command;
import org.datavec.api.split.FileSplit;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;

public class FileSplitCommand implements Command {
    @Override
    public void execute() {
        var logger = LoggerFactory.getLogger(FileSplitCommand.class);
        var allowedFormats = new String[]{".jpeg",".jpg"};
        var file = new File("data");
        var split = new FileSplit(file,allowedFormats,true);

        var msg = String.format("Found jpeg files %d",split.length());
        logger.info(msg);
        Arrays.stream(split.locations()).forEach(ur-> logger.info(ur.toString()));
    }
}
