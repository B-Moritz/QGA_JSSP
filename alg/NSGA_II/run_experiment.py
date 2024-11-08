import pdb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
import time

from base_nsga_ii import ClassicalNSGAII
from qmea import QMEA
from or_benchmark import BenchmarkCollection


log_columns = f"Problem,Candidate,Repetition,Iteration,Time,"
log_columns += f"Min Makespan,Max Makespan,Avg Makespan,Min Mean Flow Time,"
log_columns += f"Max Mean Flow Time,Avg Mean Flow Time,Spread,N Fronts,N Non-dominated solutions"

@hydra.main(version_base=None, config_name="experiment", config_path="conf")
def run_experiment(cfg: DictConfig):
    test_benchmark_collection = BenchmarkCollection(make_web_request=False)

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
        # Create log file
        cur_log_file_name = os.path.join(log_path, f"{candidate}_{cfg.experiment.experiment_id}.csv")
        if not os.path.exists(cur_log_file_name):
            # Creating the log file
            with open(cur_log_file_name, "w") as log_file:
                log_file.write(log_columns + "\n")
                
        else:
            raise Exception(f"Log file already exists: {cur_log_file_name}")
    
        for problem_name in cfg.experiment.problem_names:
            # For each problemname defined in the config run each candidate with a certain repetition
            cur_n_machines = test_benchmark_collection.benchmark_collection[problem_name]['n_machines']
            cur_n_jobs = test_benchmark_collection.benchmark_collection[problem_name]['n_jobs']
            cur_jssp_problem = test_benchmark_collection.benchmark_collection[problem_name]['problem_matrix']

            for i in range(cfg.experiment.repetitions):
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
                # Convert factory wrapper to the actual class
                candidate_obj = candidate_obj_wrapper()
                # Load the existing population object
                dump_folder = os.path.join(log_path, f"population_dumps")
                if 
                    candidate_obj.get_population()

                cur_start_time = time.time()
                for iteration_data in candidate_obj.execute():
                    time_since_start = time.time() - cur_start_time
                    with open(cur_log_file_name, "a") as log_file:
                        log_line = f"{problem_name},{candidate},{i},{iteration_data['Iteration']},{time_since_start:.4f},"
                        log_line += f"{iteration_data['Makespan']['Min']},{iteration_data['Makespan']['Max']},{iteration_data['Makespan']['Avg']},"
                        log_line += f"{iteration_data['Mean flow time']['Min']},{iteration_data['Mean flow time']['Max']},{iteration_data['Mean flow time']['Avg']},"
                        log_line += f"{iteration_data['Spread']},{iteration_data['n_fronts']},{iteration_data['n_non_dominated_solutions']}"
                        log_file.write(log_line + "\n")

                    if cfg.experiment.dump_population:
                        # Dump the population
                        candidate_obj.dump_population(dump_folder, f"{problem_name}_{candidate}_{i}_{iteration_data['Iteration']}.pkl")

                # Create gnatt diagrams
                candidate_obj.save_pareto_front_gnatt(log_path, f"{problem_name}_{candidate}_{i}")
                # Save plot of objective space
                candidate_obj.save_objectie_space_plot(log_path, f"Final_Objective_space_{problem_name}_{candidate}_{i}.png", f"{problem_name}_{candidate}_{i}")

            # Dump the population
            if not cfg.experiment.dump_population:
                candidate_obj.dump_population(dump_folder, f"{problem_name}_{candidate}_{i}_{iteration_data['Iteration']}.pkl")


if __name__=="__main__":
    run_experiment()
