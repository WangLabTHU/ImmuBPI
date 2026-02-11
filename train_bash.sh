#!/bin/bash

# iter for the fold `f` on binding
for ((i=0; i<=4; i++))
do
    python run_main.py -c ./configs/binding_config.yaml -f "$i"
done

# iter for the fold `f` on presentation
for ((i=0; i<=4; i++))
do
    python run_main.py -c ./configs/presentation_config.yaml -f "$i"
done

# iter for the fold `f` on immu
for ((i=0; i<=9; i++))
do
    python run_main.py -c ./configs/immunogenicity_config.yaml -f "$i"
done
