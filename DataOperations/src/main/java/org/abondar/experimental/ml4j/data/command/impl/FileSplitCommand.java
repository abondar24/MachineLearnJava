package org.abondar.experimental.ml4j.data.command.impl;

import org.abondar.experimental.ml4j.command.Command;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.datavec.api.split.FileSplit;

import java.io.File;
import java.util.Arrays;

public class FileSplitCommand implements Command {
    @Override
    public void execute() {
        var logger = new Slf4jLogger(FileSplitCommand.class);
        var allowedFormats = new String[]{".jpeg",".jpg"};
        var file = new File("data");
        var split = new FileSplit(file,allowedFormats,true);

        var msg = String.format("Found jpeg files %d",split.length());
        logger.info(msg);
        Arrays.stream(split.locations()).forEach(System.out::println);
    }
}
