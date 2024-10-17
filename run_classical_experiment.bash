#!/bin/bash

ls /QGA_JSSP/logdata

#python ./alg/NSGA_II/run_experiment.py -cn experiment_qga_position.yaml

for pop_size in 150 200 250
do
        echo 'Running with N=$pop_size'
        python ./alg/NSGA_II/run_experiment.py -cn experiment_classical.yaml "classical={pop_object : {N : $pop_size}}"
done