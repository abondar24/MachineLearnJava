package org.abondar.experimental.ml4j.command;

public abstract class CommandSwitcher {

    protected final CommandExecutor executor;

    public CommandSwitcher() {
        this.executor = new CommandExecutor();
    }

    public abstract void executeCommand(String cmd);
}
