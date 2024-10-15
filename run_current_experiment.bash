#!/bin/bash

ls /app/logdata

#python ./alg/NSGA_II/run_experiment.py -cn experiment_qga_position.yaml

for pop_size in 150 200 250
do
        echo 'Running with N=$pop_siz'
        python ./alg/NSGA_II/run_experiment.py -cn experiment_classical.yaml "classical={pop_object : {N : $pop_size}}"

done
