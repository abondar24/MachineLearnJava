package org.abondar.experimental.ml4j.deepnet;

import org.abondar.experimental.ml4j.command.CommandSwitcher;
import org.abondar.experimental.ml4j.deepnet.command.NetCommandSwitcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        CommandSwitcher dcs = new NetCommandSwitcher();
        if (args.length==0){
            logger.error("Missing argument. Please check documentation for available arguments");
            System.exit(0);
        }
        String cmd = args[0].toUpperCase();
        dcs.executeCommand(cmd);
    }
}
