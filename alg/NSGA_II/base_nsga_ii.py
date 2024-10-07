"""
    This file contains the NSGA-II implementation. The performance of this algorithm is used as baseline for comparing witht he Quantum GA implementations.

    References:
        * Ripon, K. S. N., Tsang, C.-H., & Kwong, S. (2007). An Evolutionary Approach for Solving the Multi-Objective Job-Shop Scheduling Problem. In Studies in Computational Intelligence (Vol. 49, pp. 165–195). https://doi.org/10.1007/978-3-540-48584-1_7

    """
import numpy as np
import pickle
import copy
import pdb
import os
from omegaconf import DictConfig

from operations import Operation
from schedules import Schedule
from individual import Individual, PermutationChromosome
from nsga_population import ClassicalPopulation

from or_benchmark import BenchmarkCollection
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
import threading



class JsspAlgorithm:
    def __init__(self, 
                 n_iterations, 
                 activate_logging=False
                ):
            self.n_iterations = n_iterations
            self.activate_logging = activate_logging

    
    def print_performance(self, cur_result: dict):
        print(f"{cur_result['Iteration']} : Makespan: [min : {cur_result['Makespan']['Min']:.2f}, avg : {cur_result['Makespan']['Avg']:.2f}], " + \
            f"Mean flow time: [min : {cur_result['Mean flow time']['Min']:.2f}, avg : {cur_result['Mean flow time']['Avg']:.2f}], " + \
                f"Spread : {cur_result['Spread']}, n_fronts: {cur_result['n_fronts']}, n_non_dominated_solutions: {cur_result['n_non_dominated_solutions']}")
    
    def dump_population(self, log_path, file_name):
        dump_folder = os.path.join(log_path, f"population_dumps")
        if not os.path.exists(dump_folder):
            os.mkdir(dump_folder)

        cur_file_path = os.path.join(dump_folder, file_name)
        with open(cur_file_path, "wb") as cur_dump_file:
            pickle.dump(self.pop_object, cur_dump_file)


    def save_pareto_front_gnatt(self, log_path, folder_name: str):
        # Save the schedules of the pareto front in the log folder
        img_folder = os.path.join(log_path, f"gnatt_imgs")
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
        
        img_folder = os.path.join(img_folder,f"{folder_name}")
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
        
        cur_pareto_front = self.pop_object.get_front_range(0)
        for i in range(cur_pareto_front[0], cur_pareto_front[1], 1):
            cur_img = self.pop_object.R[i].schedule.get_image()
            #cur_img = np.transpose(cur_img, (2, 0, 1))
            im = Image.fromarray(np.uint8(cur_img*255)).convert('RGB')
            cur_img_path = os.path.join(img_folder, f"Schedule_{i}.png")
            im.save(cur_img_path)

    def save_objectie_space_plot(self, log_path: str, figure_name: str, folder_name: str, minimum_opacity: float=0.5, color_mapper: str="plt.cm.inferno"):
        # Save a scatter plot of the pareto front in the log folder
        img_folder = os.path.join(log_path, f"objective_space_plot")
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        img_folder = os.path.join(img_folder,f"{folder_name}")
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
        
        fig, ax = plt.subplots(figsize=(10, 10))

        for i, start in enumerate(self.pop_object.front_start_index):
            print("New front: " + str(i))
            cur_front = self.pop_object.get_front_range(i)

            cur_color = list(eval(color_mapper)(((i-1)/len(self.pop_object.front_start_index))))
            cur_color[-1] = np.max([1 - i/len(self.pop_object.front_start_index), minimum_opacity])
            cur_color = tuple(cur_color)

            end = cur_front[1]
            if start == self.pop_object.N:
                # Plot the rest of the population without lines
                cur_data = np.asarray([individual.cur_fitness for individual in self.pop_object.R[self.pop_object.N:]])
                plt.scatter(cur_data[:, 0], cur_data[:, 1], color=cur_color)
            else:
                x_list = np.empty(cur_front[-1])
                y_list = np.empty(cur_front[-1])

                for j, ind in enumerate(self.pop_object.R[cur_front[0] : end]):
                    x, y = ind.cur_fitness
                    x_list[j] = x
                    y_list[j] = y

                index_sort = x_list.argsort()
                index_sort = np.flip(index_sort)
                y_list = y_list[index_sort]
                x_list = x_list[index_sort]
                #y_list.sort()
                print(x_list, y_list)
                plt.plot(x_list, y_list, color=cur_color)
                plt.scatter(x_list, y_list, color=cur_color)

        #cur_scatter = ax.scatter(point_list[:, 0], point_list[:, 1], c=color_list)
        ax.set_title(figure_name)
        #ax.set_xticks(np.linspace(np.min(point_list[:, 0])-10, np.max(point_list[:, 0])+10, 10))
        #ax.set_yticks(np.linspace(np.min(point_list[:, 1])-10, np.max(point_list[:, 1])+10, 10))
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Mean Flow Time")
        
        fig.savefig(os.path.join(img_folder, figure_name))
        plt.close()


class ClassicalNSGAII(JsspAlgorithm):

    def __init__(self,
                 n_iterations, 
                 pop_object: ClassicalPopulation,
                 activate_logging=False,
            ):
        super().__init__(n_iterations, activate_logging)

        # Initialize the population - This populates the population with randomly generated individuals
        self.pop_object = pop_object
        


    def visualize_results(self):
        plt.ion()
        fig = plt.figure(1, figsize=(15, 10))
        plt.clf()
        ax1 = fig.add_subplot(1, 2, 2)
        ax2 = fig.add_subplot(1, 2, 1)

        cur_img = self.pop_object.R[0].schedule.plot_gnatt_img(height=500, display=False, axis=ax1)
        point_list, color_list = self.pop_object.plot_fronts()
        cur_scatter = ax2.scatter(point_list[:, 0], point_list[:, 1], c=color_list)
        ax2.set_xticks(np.linspace(0, np.max(point_list[:, 0]), 10))
        ax2.set_yticks(np.linspace(0, np.max(point_list[:, 1]), 10))

        while self.n_iterations > 0:
            cur_img.set_data(self.pop_object.R[0].schedule.get_image(height=500))
            point_list, color_list = self.pop_object.plot_fronts()
            
            cur_scatter.set_offsets(point_list)
            cur_scatter.set_color(color_list)
            #ax2.scatter(point_list[:, 0], point_list[:, 1], c=color_list)
            plt.pause(0.01)

    def execute(self):
        self.pop_object.evaluate_fitness()
        self.pop_object.non_dominated_sorting()
        self.pop_object.crowding_distance_sort_all_fronts()
        max_iteration = self.n_iterations
        while self.n_iterations > 0:
            # higher tournament size -> more elitism, smaller torunament size -> less elitism
            self.pop_object.select_parents()
            self.pop_object.execute_recombination()

            # Make fitness evaluaiton
            self.pop_object.evaluate_fitness()
            self.pop_object.non_dominated_sorting()
            self.pop_object.crowding_distance_sort_all_fronts()
            cur_result = self.pop_object.get_performance()
            cur_result["Iteration"] = max_iteration - self.n_iterations
            self.print_performance(cur_result)
            
            if self.activate_logging:
                # The method will act as generator if logging is active
                yield cur_result
           
        
            self.n_iterations -= 1


if __name__=="__main__":
    problem_name = 'ft06'
    test_benchmark_collection = BenchmarkCollection(make_web_request=False)
    test_benchmark_collection.benchmark_collection[problem_name]['problem_matrix']
    n_machines = test_benchmark_collection.benchmark_collection[problem_name]['n_machines']
    n_jobs = test_benchmark_collection.benchmark_collection[problem_name]['n_jobs']
    jssp_problem = test_benchmark_collection.benchmark_collection[problem_name]['problem_matrix']
    n_iterations = 1000

    optim = ClassicalNSGAII(n_iterations, n_jobs, n_machines, jssp_problem)
    optim.nsga2(show_gui=False)
    