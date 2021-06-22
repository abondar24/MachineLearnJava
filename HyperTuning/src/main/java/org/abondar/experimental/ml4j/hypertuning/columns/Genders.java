package org.abondar.experimental.ml4j.hypertuning.columns;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public enum Genders {
    Male,
    Female;

    public static List<String> getList(){
        return Stream.of(Genders.values())
                .map(Genders::name)
                .collect(Collectors.toList());

    }
}
