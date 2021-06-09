package org.abondar.experimental.ml4j.data.command.reader;

import org.abondar.experimental.ml4j.command.Command;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class ImageReaderCommand implements Command {
    @Override
    public void execute() {
        var logger = LoggerFactory.getLogger(ImageReaderCommand.class);

        var imgHeight = 100;
        var imgWidth = 100;
        var channels = 3;

        var labelGenerator = new ParentPathLabelGenerator();
        var file = new File("data/images/n02131653_124.jpeg");
        var split = new FileSplit(file);

        try {
            var imageReader = new ImageRecordReader(imgHeight,imgWidth,channels,labelGenerator);
            imageReader.initialize(split);

            while (imageReader.hasNext()){
                imageReader.nextRecord()
                        .getRecord()
                        .forEach(writable -> logger.info(writable.toString()));
            }

        } catch (IOException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }


    }
}
