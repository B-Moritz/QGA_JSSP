import pdb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
import time
import re
import subprocess

from base_nsga_ii import ClassicalNSGAII
from qmea import QMEA
from or_benchmark import BenchmarkCollection


@hydra.main(version_base=None, config_name="experiment", config_path="conf")
def run_experiment(cfg: DictConfig):
    test_benchmark_collection = BenchmarkCollection(reload_benchmarks=False)

    if cfg.experiment.checkpoint.continue_from_chekcpoint:
        # Use existing log folder
        # Check that the specified log folder exists
        log_path = cfg.experiment.checkpoint.log_folder_path
        if not os.path.exists(log_path):
            raise FileNotFoundError("Could not find the specified log folder directory")
    else:
        # Create log folder
        par_dir = os.path.dirname(str(__file__))
        print(par_dir)
        cur_timestamp = str(datetime.datetime.today()).replace(' ', '_').replace(".", "_").replace(":", "_")
        logdata_path = os.path.join(*cfg.experiment.log_path)
        log_path = os.path.join(par_dir, logdata_path + f"_{cfg.experiment.experiment_id}_{cur_timestamp}")
        if not os.path.exists(log_path):
            # Create the log folder
            os.mkdir(log_path)
            print("Created log direcotry at: " + str(log_path))
        else:
            raise Exception("Log folder already exists. Please find a different name for this experiment")

    # Copy current config to the log folder    
    target = os.path.join(log_path, "experiment.yaml")
    OmegaConf.save(cfg, target)
    
    # Create the list of candidate algorithms    
    candidate_list = dir(cfg)
    candidate_list.remove("experiment")



    for candidate in candidate_list:
        # Extract objectives
        objective_1, objective_2 = eval(f"cfg.{candidate}.pop_object.objectives") 
        # Create log file
        log_columns = f"Problem,Candidate,Repetition,Iteration,Time,"
        log_columns += f"Min {objective_1},Max {objective_1},Avg {objective_1},Min {objective_2},"
        log_columns += f"Max {objective_2},Avg {objective_2},Spread,N Fronts,N Non-dominated solutions"

        cur_log_file_name = os.path.join(log_path, f"{candidate}_{cfg.experiment.experiment_id}.csv")
        if not os.path.exists(cur_log_file_name):
            # Creating the log file
            with open(cur_log_file_name, "w") as log_file:
                log_file.write(log_columns + "\n")
                
        else:
            raise Exception(f"Log file already exists: {cur_log_file_name}")
    
        for problem_name in cfg.experiment.problem_names:
            # For each problem name defined in the config run each candidate with a certain repetition
            cur_n_machines = test_benchmark_collection.benchmark_collection[problem_name]['n_machines']
            cur_n_jobs = test_benchmark_collection.benchmark_collection[problem_name]['n_jobs']
            cur_jssp_problem = test_benchmark_collection.benchmark_collection[problem_name]['problem_matrix']

            for i in range(cfg.experiment.repetitions):
                # Re run algorithm instance
                # Instantiate candidate
                print(f"\n\nProblem {problem_name}, Candidate {candidate}, Repetition {i}", end="\n\n")
                candidate_obj_wrapper = hydra.utils.instantiate(cfg[candidate],
                                                        n_iterations=cfg.experiment.n_iterations,
                                                        pop_object={
                                                            "n_jobs" : cur_n_jobs,
                                                            "n_machines" : cur_n_machines,
                                                            "jssp_problem" : cur_jssp_problem
                                                        }
                                                    )
                # Convert factory wrapper to the actual class which is one of the algorithm objects defined in base_nsga_ii
                candidate_obj = candidate_obj_wrapper()
                # Load the existing population object
                dump_folder = os.path.join(log_path, f"population_dumps")
                # Continue from checkpoint if flagged in config
                if cfg.experiment.checkpoint.continue_from_chekcpoint:
                    # Get all file names
                    cur_dump_file_pattern = f"{problem_name}_{candidate}_{i}_(\d+).pkl"
                    file_names = os.listdir(dump_folder)
                    cur_max_iteration = 0
                    for cur_file in file_names:
                        cur_match = re.match(cur_dump_file_pattern, cur_file).groups()[0]
                        if cur_max_iteration < int(cur_match):
                            cur_max_iteration = int(cur_match)
                    # Load population
                    candidate_obj.get_population(dump_folder, f"{problem_name}_{candidate}_{i}_{cur_max_iteration}.pkl")

                cur_start_time = time.time()
                for iteration_data in candidate_obj.execute():
                    time_since_start = time.time() - cur_start_time
                    with open(cur_log_file_name, "a") as log_file:
                        log_line = f"{problem_name},{candidate},{i},{iteration_data['Iteration']},{time_since_start:.4f},"
                        log_line += f"{iteration_data[objective_1]['Min']},{iteration_data[objective_1]['Max']},{iteration_data[objective_1]['Avg']},"
                        log_line += f"{iteration_data[objective_2]['Min']},{iteration_data[objective_2]['Max']},{iteration_data[objective_2]['Avg']},"
                        log_line += f"{iteration_data['Spread']},{iteration_data['n_fronts']},{iteration_data['n_non_dominated_solutions']}"
                        log_file.write(log_line + "\n")

                    if cfg.experiment.dump_population:
                        # Dump the population
                        candidate_obj.dump_population(dump_folder, f"{problem_name}_{candidate}_{i}_{iteration_data['Iteration']}.pkl")

                # Create gnatt diagrams
                candidate_obj.save_pareto_front(log_path, f"{problem_name}_{candidate}_{i}")
                # Save plot of objective space
                candidate_obj.save_objective_space_plot(log_path, f"Final_Objective_space_{problem_name}_{candidate}_{i}.png", f"{problem_name}_{candidate}_{i}")

            # Dump the population if it was not done before
            if not cfg.experiment.dump_population:
                candidate_obj.dump_population(dump_folder, f"{problem_name}_{candidate}_{i}_{iteration_data['Iteration']}.pkl")


if __name__=="__main__":
    run_experiment()
