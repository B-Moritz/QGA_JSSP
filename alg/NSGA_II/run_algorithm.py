import pdb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
import time
import re
import numpy as np
import copy

import tkinter as tk
import threading
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from matplotlib.pyplot import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from functools import *

from base_nsga_ii import ClassicalNSGAII
from qmea import QMEA
from or_benchmark import BenchmarkCollection

class SharedPopulation:
    pop_object = None
    iteration_counter = 0
    previous_counter = 0

def exception_handler(func):
    def wrapper_exception_handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)
            return
        
    return wrapper_exception_handler

def update_objective_space_plot(frames, ax, shared_obj):
    #while True:
    #    # Listen to progress in the main algorithm routine
    #    if shared_obj.previous_counter - shared_obj.iteration_counter > 0:
    #        break

    minimum_opacity: float = 0.5
    color_mapper: str = "cm.inferno"
    

    if shared_obj.pop_object:
        ax.clear()
        ax.set_title(f"Objective Space iter: {shared_obj.iteration_counter}")        
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Mean Flow Time")
        cur_pop_object = copy.deepcopy(shared_obj.pop_object)
        for i, start in enumerate(cur_pop_object.front_start_index):
            #print("New front: " + str(i))

            cur_front = cur_pop_object.get_front_range(i)

            #print("front_start_index: " + str(len(shared_obj.pop_object.front_start_index)))
            cur_color = list(eval(color_mapper)(((i-1)/len(cur_pop_object.front_start_index))))
            cur_color[-1] = np.max([1 - i/len(cur_pop_object.front_start_index), minimum_opacity])
            cur_color = tuple(cur_color)

            end = cur_front[1]
            if start == shared_obj.pop_object.N:
                # Plot the rest of the population without lines
                cur_data = np.asarray([individual.cur_fitness for individual in cur_pop_object.R[cur_pop_object.N:]])
                ax.scatter(cur_data[:, 0], cur_data[:, 1], color=cur_color)
            else:
                x_list = np.empty(cur_front[-1])
                y_list = np.empty(cur_front[-1])

                for j, ind in enumerate(cur_pop_object.R[cur_front[0] : end]):
                    x, y = ind.cur_fitness
                    x_list[j] = x
                    y_list[j] = y

                index_sort = x_list.argsort()
                index_sort = np.flip(index_sort)
                y_list = y_list[index_sort]
                x_list = x_list[index_sort]
                #y_list.sort()
                #print(x_list, y_list)
                ax.plot(x_list, y_list, color=cur_color)
                ax.scatter(x_list, y_list, color=cur_color)



@hydra.main(version_base=None, config_name="experiment", config_path="conf")
def run_algorithm(cfg: DictConfig):
    
    # Create the gui
    # the figure that will contain the plot 
    fig, ax = subplots(1, 1, figsize = (5, 5), 
                    dpi = 100) 
    
    def _quit():
        window.quit()
        window.destroy() 

    class ContinueFlag():
        continue_flag = True


    shared_obj = SharedPopulation()
    # the main Tkinter window 
    window = tk.Tk() 
    window.protocol("WM_DELETE_WINDOW", _quit)
    # setting the title  
    window.title('Plotting in Tkinter') 
    
    # dimensions of the main window 
    window.geometry("700x700") 

    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
                                master = window)   
    canvas.draw() 

    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack()
    continue_flag = ContinueFlag()
    alg_thread = threading.Thread(target=algorithm_thread, args=(cfg, continue_flag, shared_obj))
    alg_thread.start()

    animator = animation.FuncAnimation(fig, partial(update_objective_space_plot, ax=ax, shared_obj=shared_obj), interval=500, cache_frame_data=False)

    tk.mainloop()
    continue_flag.continue_flag = False
    print("Avaiting termination of algorithm thread.")
    alg_thread.join()

#@exception_handler
def algorithm_thread(cfg: DictConfig, continue_flag, shared_obj):
    test_benchmark_collection = BenchmarkCollection(reload_benchmarks=False)
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
    
        for problem_name in cfg.experiment.problem_names:
            # For each problemname defined in the config run each candidate with a certain repetition
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
                
                for iteration_data in candidate_obj.execute():
                    if not continue_flag.continue_flag:
                        # Terminate execution if the stop flag is raised
                        return
                    
                    shared_obj.pop_object = copy.deepcopy(candidate_obj.pop_object)
                    shared_obj.iteration_counter += 1

                    
                # Create gnatt diagrams
                #candidate_obj.save_pareto_front_gnatt(log_path, f"{problem_name}_{candidate}_{i}")
                # Save plot of objective space
                

            # Dump the population if it was not done before
            #if not cfg.experiment.dump_population:
            #    candidate_obj.dump_population(dump_folder, f"{problem_name}_{candidate}_{i}_{iteration_data['Iteration']}.pkl")


if __name__=="__main__":
    run_algorithm()
