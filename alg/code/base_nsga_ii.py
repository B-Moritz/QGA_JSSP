"""
    This file contains the NSGA-II implementation. The performance of this algorithm is used as baseline for comparing witht he Quantum GA implementations.

    References:
        * Ripon, K. S. N., Tsang, C.-H., & Kwong, S. (2007). An Evolutionary Approach for Solving the Multi-Objective Job-Shop Scheduling Problem. In Studies in Computational Intelligence (Vol. 49, pp. 165–195). https://doi.org/10.1007/978-3-540-48584-1_7

    """
import numpy as np
import copy
import pdb

from operations import Operation
from schedules import Schedule
from individual import Individual
from or_benchmark import BenchmarkCollection
import matplotlib.pyplot as plt

from typing import List

class Population:
    # Consists of the parent and ofspring population P_t and Q_t to form the full population R_t
    def __init__(self, 
                 N: int, 
                 decoding_method: str, 
                 n_jobs: int, 
                 n_machines: int, 
                 jssp_problem: np.ndarray,
                 activate_schedule: bool = False,
                 objectives: List[str] = ["makespan", "mean_flow_time"],
                 population_array: np.ndarray = np.array([])
                ):
        # Define the total population size
        self.N = N
        self.n_jobs: int = n_jobs
        self.n_machines: int = n_machines
        self.jssp_problem: np.ndarray = jssp_problem
        self.decoding_method: str = decoding_method
        self.activate_schedule: bool = activate_schedule
        self.objectives: List[str] = objectives

        if len(population_array) == 0:
            self.initialize_population()
        else:
            self.R = population_array

    def initialize_population(self):
        # P is index 0 - N-1, while Q is index N - 2N
        self.R = np.empty(2*self.N, dtype=Individual)
        for i in range(len(self.R)):
            self.R[i] = Individual.create_individual(self.n_jobs, self.n_machines)

    def evaluate_fitness(self):
        for individual in self.R:
            individual.create_schedule(
                self.decoding_method, 
                self.jssp_problem, 
                self.objectives,
                self.activate_schedule
            )
        # Fintess values are available as self.R[i].schedule.max_completion_time
            

    def select_parents(self, tournament_size, mating_pool_size):
        self.mating_pool = np.empty(mating_pool_size, dtype=Individual)
        for selected_index in range(len(self.mating_pool)):
            # Find the smallest index among the integers representing the index of the array R sorted by non-dominated sorting and then crowding distance
            cur_index = np.min(np.random.randint(0, self.N, tournament_size))
            self.mating_pool[selected_index] = self.R[cur_index]
        
    def gox_crossover(self, donator: np.ndarray, receiver: np.ndarray):
        """This method performs the Generalized Order Crossover described in (Ripon et al., 2007).

        Parameters
        ----------
        donator : np.ndarray
            The donator genotype
        receiver : np.ndarray
            The receiver genotype

        Returns
        -------
        Individual
            The individual object for the produced child.
        """
        # Define the range in the parent genotype that should be extracted
        start_range = np.random.randint(0, len(donator))
        section = np.random.uniform() * (0.5-0.3) + 0.3 # Random value between 0.3 and 0.5
        lenght_of_range = int(np.floor(section*len(donator)))
        stop_range = int(start_range + lenght_of_range) % len(donator)
        # Changes are made to the receiver. It is therefore copied here to avoid sideeffects for the method
        copy_receiver = copy.deepcopy(receiver)
        # The child individual is created by modifying the receiver genotype 
        # First find all the genes that correspond to the genes in the donator are removed by setting them to np.inf 
        for j in range(self.n_jobs):
            # For each job - iterate over the receiver 
            swap_counter = 0
            for k in range(len(donator)):
                # For each job_index in m1 check for match in receiver
                if donator[k] == j:
                    # If the current job in the donator corresponds to the current job - find the 
                    matched = False
                    while not matched:
                        if copy_receiver[swap_counter] == j:
                            # Found match
                            matched = True
                            if ((start_range < stop_range) and (k >= start_range and k < stop_range)) \
                                or ((start_range > stop_range) and not (k < start_range and k >= stop_range)):
                                # The match is in range and should be deleted    
                                copy_receiver[swap_counter] = -1
                            
                        swap_counter += 1

        # Find the injection position
        injection_index = np.where(copy_receiver == -1)[0][0]
        child_chromosome = np.empty(len(copy_receiver), dtype=int)
        # Need to distinguish between wrapper and normal case
        if start_range > stop_range:
            # Wrapper case
            # First insert the code from donor into the child
            child_chromosome[:stop_range] = donator[:stop_range]
            child_chromosome[start_range:] = donator[start_range:]
            # Now the part from receiver 
            try:
                child_chromosome[stop_range:start_range] = copy_receiver[copy_receiver != -1]
            except Exception as e:
                print("Wrapper case")
                print(e)
                pdb.set_trace()
        else:
            # Normal case
            # Inject the section 
            child_chromosome[injection_index: injection_index + lenght_of_range] = donator[start_range:stop_range]
            # Inject the first numbers to the right of slice
            child_chromosome[:injection_index] = copy_receiver[:injection_index]
            # insert the remaining numbers from receiver to the right of slice that are not -1
            try:
                child_chromosome[injection_index + lenght_of_range:] = receiver[copy_receiver != -1][injection_index:]
            except Exception as e:
                print("Normal case")
                print(e)
                pdb.set_trace()

        return Individual(child_chromosome, self.n_jobs, self.n_machines) 

    def execute_recombination(self):
        # performs crossover on the mating_pool to fill the second half of the population
        offspring_index = self.N
        for m_index in range(0, len(self.mating_pool), 2):
            donator = self.mating_pool[m_index].genotype
            receiver = self.mating_pool[m_index + 1].genotype
            self.R[offspring_index] = self.gox_crossover(donator, receiver)
            # perform mutation on child      
            self.R[offspring_index].genotype = self.job_pair_mutation(self.R[offspring_index].genotype)

    def job_pair_mutation(self, cur_genotype: np.ndarray) -> np.ndarray:
        # job-pair exchange mutation operator (Ripon et al., 2007)

        # Picks two random genes i and j, where i != j
        # Need two random numbers - first should be between 0 and length of gene (m x j)
        first_job_index = np.random.randint(0, self.n_jobs*self.n_machines)
        # Second needs to be betweeen 0 and m x (j-1)
        second_job_index = np.random.randint(0, (self.n_jobs-1)*self.n_machines)
        first_job_num = cur_genotype[first_job_index]
        # iterate over the genotype to count the genes that are not corresponding to first job number
        counter = 0
        for i in range(len(cur_genotype)):
            if first_job_num != cur_genotype[i] and counter == second_job_index:
                second_index_converted = i
                break
            elif first_job_num != cur_genotype[i] and counter < second_job_index:
                counter += 1


        # Make the swap
        cur_genotype[first_job_index] = cur_genotype[second_index_converted]
        cur_genotype[second_index_converted] = first_job_num
        return cur_genotype


    def non_dominated_sorting(self):
        # Sort the entire population
        self.front_start_index: List = [] # Contains start index for each front
        cur_start = 0
        cur_swap_index = 0
        while cur_start < self.N:
            for i in range(cur_start, len(self.R)):
                # For each individual go through the rest to find the best posible individual (that dominates all solutions)
                dominated = False
                for j in range(cur_start, len(self.R)):
                    # swap if J dominates the 
                    if (i != j) and (np.array(self.R[j].cur_fitness) <= np.array(self.R[i].cur_fitness)).all():
                        # Break the loop if the individual i is dominated -> means that the individual does not belong to the current front
                        dominated = True
                        break
                
                if not dominated: 
                    # The individual was not dominated -> place it in the front by doing a swap 
                    self.swap(cur_swap_index, i)
                    # Move swap position one step to the right
                    cur_swap_index += 1

            # What happens when all solutions are dominated. 
            # What happens if only one individual is in a front

            # Close the front 
            if cur_start == cur_swap_index:
                # There is only one front in the set, 
                self.front_start_index.append(cur_start)
                cur_start = len(self.R)
            else:
                # Set the front range and the next start index one step to the right of the range 
                self.front_start_index.append(cur_start)
                cur_start = cur_swap_index

    def get_front_range(self, i: int) -> List: # Returns list with [start_index, stop_index+1, length_of_front]
        if i + 1 >= len(self.front_start_index):
            cur_length = len(self.R) - self.front_start_index[i]
            return([self.front_start_index[i], len(self.R), cur_length])
        else:
            cur_length = self.front_start_index[i+1] - self.front_start_index[i]
            return([self.front_start_index[i], self.front_start_index[i+1], cur_length])
        
    def get_performance(self) -> np.ndarray:
        """This method is used to calculate the performance metrics for the non-dominated front

        Returns
        -------
        np.ndarray
            Array containing the metrics: makespan [min, avg], mean flow time [min, avg], spread
        """
        cur_range = self.get_front_range(0)
        result = {"Makespan": {"Avg" : 0, "Min": np.inf,}, "Mean flow time": {"Avg" : 0, "Min": np.inf}, "Spread" : np.inf}
        for i in range(cur_range[1]):
            result["Makespan"]["Avg"] += self.R[i].schedule.max_completion_time
            if result["Makespan"]["Min"] > self.R[i].schedule.max_completion_time:
                result["Makespan"]["Min"] = self.R[i].schedule.max_completion_time

            result["Mean flow time"]["Avg"] += self.R[i].schedule.mean_flow_time
            if result["Mean flow time"]["Min"] > self.R[i].schedule.mean_flow_time:
                result["Mean flow time"]["Min"] = self.R[i].schedule.mean_flow_time

        result["Makespan"]["Avg"] = result["Makespan"]["Avg"] / cur_range[2]
        result["Mean flow time"]["Avg"] = result["Mean flow time"]["Avg"] / cur_range[2]
        return result


    def crowding_distance_sort_all_fronts(self):
        """This method is used to perform crowding distance sorting on all fronts in the population
        """
        for i in range(len(self.front_start_index)):
            # Sort every front on chrowding distance
            cur_front = self.get_front_range(i)
            self.crowding_distance_sort_front(cur_front)


    def swap(self, i, j):
        temp = self.R[i]
        self.R[i] = self.R[j]
        self.R[j] = temp

    def crowding_distance_sort_front(self, cur_range):
        # This method is used to perform crowding distance sorting on the last non-dominated front in self.start_stop_fronts.
        # Needs to be executed after non-dominated sorting
        # As a result the last front is sorted and the population is ready for parent selection and recombination
        #cur_range = self.start_stop_fronts[-1]
        range_len = cur_range[2]
        order_list = []
        for i, individual in enumerate(self.R[cur_range[0] : cur_range[1]]):
            cur_order_list = [i]
            cur_order_list.extend([getattr(individual.schedule, objective) for objective in self.objectives])
            cur_order_list.append(0)
            order_list.append(cur_order_list)

        for i in range(1, len(self.objectives)+1):
            # Order the indexes accordingt to objective i
            order_list.sort(key=lambda x: x[i])
            # Set boundaries to infinity
            order_list[0][-1] = np.inf
            order_list[-1][-1] = np.inf
            for j in range(1, len(order_list)-1):
                # Increment distance for current individual
                order_list[j][-1] += (order_list[j+1][i] - order_list[j-1][i])/(order_list[-1][i] - order_list[0][i])

        # Sort on distance
        order_list.sort(key=lambda x: x[-1], reverse=True)
        # Reflect the new order on the population    
        temp_R = np.empty(range_len, dtype=Individual)
        for i, k in enumerate(order_list):
            temp_R[i] = self.R[cur_range[0] + k[0]]

        self.R[cur_range[0] : cur_range[1]] = temp_R
    
    def plot_fronts(self):
        # This method produces a scatter plot for the different fronts
        
        color_list = np.ones((len(self.R), 4))
        point_list = np.zeros((len(self.R), 2))

        for i in range(len(self.front_start_index)):
            cur_range = self.get_front_range(i)
            for j in range(cur_range[0], cur_range[1], 1):
                point_list[j, :] = self.R[j].cur_fitness
                color_list[j, :] = plt.cm.gist_rainbow(i/len(self.front_start_index)) # viridis
                
        return [point_list, color_list]

def nsga2(n_jobs, n_machines, jssp_problem, n_iterations, gui=False):
    # Initialize the population - This populates the population with randomly generated individuals
    pop_object = Population(
                N=20, 
                decoding_method = "apply_operation_based_bierwirth", 
                n_jobs = n_jobs, 
                n_machines = n_machines, 
                jssp_problem = jssp_problem,
                activate_schedule = True,
                objectives = ["max_completion_time", "mean_flow_time"])
    
    if gui:
        plt.ion()
        fig = plt.figure(1, figsize=(15, 10))
        plt.clf()
        ax1 = fig.add_subplot(1, 2, 2)
        ax2 = fig.add_subplot(1, 2, 1)
    pop_object.evaluate_fitness()
    pop_object.non_dominated_sorting()
    pop_object.crowding_distance_sort_all_fronts()

    if gui:
        cur_img = pop_object.R[0].schedule.plot_gnatt_img(height=500, display=False, axis=ax1)
        point_list, color_list = pop_object.plot_fronts()
        cur_scatter = ax2.scatter(point_list[:, 0], point_list[:, 1], c=color_list)

    for i in range(1, n_iterations, 1):
        # higher tournament size -> more elitism, smaller torunament size -> less elitism
        pop_object.select_parents(2, 20)
        pop_object.execute_recombination()

        # Make fitness evaluaiton
        pop_object.evaluate_fitness()
        pop_object.non_dominated_sorting()
        pop_object.crowding_distance_sort_all_fronts()
        cur_result = pop_object.get_performance()
        print(f"{i} : Makespan: [min : {cur_result['Makespan']['Min']:.2f}, avg : {cur_result['Makespan']['Avg']:.2f}], " + \
              f"Mean flow time: [min : {cur_result['Mean flow time']['Min']:.2f}, avg : {cur_result['Mean flow time']['Avg']:.2f}]")
        
        if gui:
            cur_img.set_data(pop_object.R[0].schedule.get_image(height=500))
            point_list, color_list = pop_object.plot_fronts()
            #ax2.clear()
            cur_scatter.set_offsets(point_list)
            cur_scatter.set_color(color_list)
            #ax2.scatter(point_list[:, 0], point_list[:, 1], c=color_list)
            plt.pause(0.01)

    # Plot gnatt for one of the solutions
    #pop_object.R[0].schedule.plot_gnatt_img()


if __name__=="__main__":
    test_benchmark_collection = BenchmarkCollection(make_web_request=False)
    test_benchmark_collection.benchmark_collection['ft06']['problem_matrix']
    n_machines = test_benchmark_collection.benchmark_collection['ft06']['n_machines']
    n_jobs = test_benchmark_collection.benchmark_collection['ft06']['n_jobs']
    jssp_problem = test_benchmark_collection.benchmark_collection['ft06']['problem_matrix']
    n_iterations = 100

    nsga2(n_jobs, n_machines, jssp_problem, n_iterations)