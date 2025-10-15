#!/bin/bash

ls -la /QGA_JSSP/logdata_3



#python ./alg/NSGA_II/run_experiment.py -cn experiment_qga_position.yaml

for pop_size in 50
do
        echo 'Running with N=$pop_size'
        python ./alg/NSGA_II/run_experiment.py -cn experiment_rkQMGA.yaml "quantum_position_encoding_restricted={pop_object : {N : $pop_size}}"
done
