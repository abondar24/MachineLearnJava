package org.abondar.experimental.ml4j.nlp;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.abondar.experimental.ml4j.nlp.command.NlpCommandSwitcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Main {
    private static final Logger logger =  LoggerFactory.getLogger(Main.class);
    public static void main(String[] args) {

        CommandSwitcher ncs = new NlpCommandSwitcher();
        if (args.length==0){
            logger.error("Missing argument. Please check documentation for available arguments");
            System.exit(0);
        }
        String cmd = args[0].toUpperCase();
        ncs.executeCommand(cmd);
    }
}
