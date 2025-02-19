#!/bin/bash

ls -la /QGA_JSSP/logdata_2

#python ./alg/NSGA_II/run_experiment.py -cn experiment_qga_position.yaml

for pop_size in 50
do
        echo 'Running with N=$pop_size'
        python ./alg/NSGA_II/run_experiment.py -cn experiment_hash_based.yaml "quantum_hash_base_encoding={pop_object : {N : $pop_size}}"
done
