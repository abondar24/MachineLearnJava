package org.abondar.experimental.ml4j.nlp;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.abondar.experimental.ml4j.nlp.command.NlpCommandSwitcher;
import org.bytedeco.javacpp.tools.Slf4jLogger;

public class Main {
    public static void main(String[] args) {
        var logger = new Slf4jLogger(Main.class);
        CommandSwitcher ncs = new NlpCommandSwitcher();
        if (args.length==0){
            logger.error("Missing argument. Please check documentation for available arguments");
            System.exit(0);
        }
        String cmd = args[0].toUpperCase();
        ncs.executeCommand(cmd);
    }
}