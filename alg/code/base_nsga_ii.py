# This file contains the NSGA-II implementation. The performance of this algorithm is used as baseline for comparing witht he Quantum GA implementations.
import numpy as np
import copy

from operations import Operation
from schedules import Schedule
from individual import Individual

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
        
    def gox_crossover(self):
        # performs crossover on the mating_pool to fill the second half of the population
        offspring_index = self.N
        for m_index in range(0, len(self.mating_pool, 2)):
            m1 = self.mating_pool[m_index].genotype
            m2 = self.mating_pool[m_index + 1].genotype
            start_range = np.random.randint(0, len(m1), dtype=Individual)
            section = np.random.uniform() * (0.5-0.3) + 0.3 # Random value between 0.3 and 0.5
            lenght_of_range = np.floor(section*len(m1))
            stop_range = start_range + lenght_of_range

            for j in range(self.n_jobs):
                # For each job
                swap_counter = 0
                for k in range(len(m1)):
                    # For each job_index in m1 check for match in m2
                    if m1[k] == j:
                        matched = False
                        while not matched:
                            if m2[swap_counter] == m1[k]:
                                # Found match
                                matched = True
                                if ((start_range < stop_range) and (k >= start_range and k <= stop_range)) \
                                    or ((start_range > stop_range) and (k <= start_range and k >= stop_range)):
                                    # The match is in range and should be deleted    
                                    m2[swap_counter] = np.inf
                                
                            swap_counter += 1

            # Find the injection position
            injection_index = np.where(m2 == np.inf)[0][0]
            child_chromosome = np.empty(len(m2), dtype=int)
            # Need to distinguish between wrapper and normal case
            if start_range > stop_range:
                # Wrapper case
                # First insert the code from donor into the child
                child_chromosome[:stop_range] = m1[:stop_range]
                child_chromosome[start_range:] = m1[start_range:]
                # Now the part from receiver 
                child_chromosome[stop_range:start_range] = m1[m1 != np.inf]
            else:
                # Normal case
                # Inject the section 
                child_chromosome[injection_index: injection_index + stop_range - start_range] = m1[start_range:stop_range]
                child_chromosome[:injection_index] = m2[:injection_index]
                child_chromosome[injection_index + stop_range - start_range:] = m2[m2 != np.inf][injection_index:]
        
            self.R[offspring_index] = Individual(child_chromosome, self.n_jobs, self.n_machines)            


    def non_dominated_sorting(self):
        # Sort the entire population
        self.start_stop_fronts = [] # Contains start and stop index pairs for each front discovered in R
        cur_start = 0
        cur_swap_index = 0
        while cur_start < self.N:
            for i in range(cur_start, len(self.R)):
                # For each individual go through the rest to find the best posible individual (that dominates all solutions)
                dominated = False
                for j in range(cur_start, len(self.R)):
                    # swap if J dominates the 
                    if (i != j) and (np.array(self.R[j].cur_fitness) <= np.array(self.R[i].cur_fitness)).all():
                        # Each time the 
                        dominated = True
                        break
                
                if not dominated: 
                    self.swap(cur_swap_index, i)
                    cur_swap_index += 1

            # Close the front 
            self.start_stop_fronts.append([cur_start, cur_swap_index - 1])
            cur_start = cur_swap_index

    def get_crowding_distance_ranks(self):
        for front in self.start_stop_fronts:
            # Sort every front on chrowding distance
            self.get_crowding_distance(front)


    def swap(self, i, j):
        temp = self.R[i]
        self.R[i] = self.R[j]
        self.R[j] = temp

    def get_crowding_distance(self, cur_range):
        # This method is used to perform crowding distance sorting on the last non-dominated front in self.start_stop_fronts.
        # Needs to be executed after non-dominated sorting
        # As a result the last front is sorted and the population is ready for parent selection and recombination
        #cur_range = self.start_stop_fronts[-1]
        range_len = cur_range[1] - cur_range[0]
        order_list = []
        for i, individual in enumerate(self.R[cur_range[0] : cur_range[1]+1]):
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

        self.R[cur_range[0] : cur_range[1]+1] = temp_R


            
        
    

#def nsga2():


if __name__=="__main__":
    # Test the none dominated sorting
    test_pop = Population()