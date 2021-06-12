package org.abondar.experimental.ml4j.data.command.split;

import org.abondar.experimental.ml4j.command.Command;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.TransformSplit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;

public class TransformSplitCommand implements Command {

    private static final Logger logger = LoggerFactory.getLogger(TransformSplitCommand.class);

    @Override
    public void execute() {

        try {

            TransformSplit.URITransform uriTransform = URI::normalize;
            List<URI> uris = List.of(
                    new URI("file:/home/abondar/IdeaProjects/MachineLearnJava/DataOperations/data/./a-in.txt"),
                    new URI("file:/home/abondar/IdeaProjects/MachineLearnJava/DataOperations/data//b-in.txt"),
                    new URI("file:/home/abondar/IdeaProjects/MachineLearnJava/DataOperations/data/c-in.txt"));

            logger.info("Incoming Uris");
            uris.forEach(ur-> logger.info(ur.toString()));

            var cis = new CollectionInputSplit(uris);
            var nts = new TransformSplit(cis,uriTransform);

            logger.info("Normalized incoming URIs");
            Arrays.stream(nts.locations()).forEach(ur-> logger.info(ur.toString()));

            logger.info("Transformed to out URIs");
            var ts = TransformSplit.ofSearchReplace(nts,"-in.txt","-out.txt");
            Arrays.stream(ts.locations()).forEach(ur-> logger.info(ur.toString()));



        }catch (URISyntaxException ex){
            logger.error(ex.getMessage());
            System.exit(2);
        }


    }
}
