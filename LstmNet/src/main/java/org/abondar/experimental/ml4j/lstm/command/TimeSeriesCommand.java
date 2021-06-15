package org.abondar.experimental.ml4j.lstm.command;

import org.abondar.experimental.ml4j.command.Command;
import org.abondar.experimental.ml4j.lstm.net.Physionet;

public class TimeSeriesCommand implements Command {

    @Override
    public void execute() {
          var net = new Physionet();
          net.runComputations();
    }


}
