#!/bin/bash
#echo "Hello World"
#dvc remote add -d -f googleremote gdrive://${{ secrets.GOOGLE_DRIVE_URI }}
#dvc remote modify myremote gdrive_acknowledge_abuse true
#dvc remote modify googleremote gdrive_use_service_account true
#dvc remote modify googleremote --local gdrive_service_account_json_file_path creds.json

python ./alg/NSGA_II/run_experiment.py -cn experiment_1.yaml