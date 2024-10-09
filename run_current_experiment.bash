#!/bin/bash

ls /app/logdata

python ./alg/NSGA_II/run_experiment.py -cn experiment_qga_position.yaml
