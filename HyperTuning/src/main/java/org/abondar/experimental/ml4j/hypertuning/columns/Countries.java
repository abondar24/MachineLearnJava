package org.abondar.experimental.ml4j.hypertuning.columns;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public enum Countries {
    France,
    Germany,
    Spain;

    public static List<String> getList(){
        return Stream.of(Countries.values())
                .map(Countries::name)
                .collect(Collectors.toList());

    }
}
