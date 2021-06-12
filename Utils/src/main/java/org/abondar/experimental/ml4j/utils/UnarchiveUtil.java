package org.abondar.experimental.ml4j.utils;

import org.apache.ant.compress.taskdefs.Unzip;
import org.codehaus.plexus.archiver.tar.TarGZipUnArchiver;
import org.codehaus.plexus.logging.console.ConsoleLoggerManager;

import java.io.File;

public class UnarchiveUtil {


   public static void unarchiveTar(File archive) {
        var unarchiver = new TarGZipUnArchiver();
        var manager = new ConsoleLoggerManager();

        manager.initialize();
        var plexusLogger = manager.getLoggerForComponent("word2vec");

        unarchiver.enableLogging(plexusLogger);
        unarchiver.setSourceFile(archive);

        unarchiver.setDestDirectory(new File("data"));
        unarchiver.extract();
    }

    public static void unarchiveZip(File zip) {
        var unzip = new Unzip();

        unzip.setSrc(zip);
        unzip.setDest(new File("data"));
        unzip.execute();
    }


    private UnarchiveUtil(){}
}
