package org.abondar.experimental.ml4j.lstm.command;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.lstm.net.UciNet;

public class SequenceDataCommand implements Command {
    @Override
    public void execute() {

        var uci = new UciNet();
        uci.runComputations();

    }
}
