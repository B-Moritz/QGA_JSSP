


"""
    This file contains the population objects used for the nsga-ii algorithm.

    References:
        * Ripon, K. S. N., Tsang, C.-H., & Kwong, S. (2007). An Evolutionary Approach for Solving the Multi-Objective Job-Shop Scheduling Problem. In Studies in Computational Intelligence (Vol. 49, pp. 165–195). https://doi.org/10.1007/978-3-540-48584-1_7
        * Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197. IEEE Transactions on Evolutionary Computation. https://doi.org/10.1109/4235.996017
        * Bierwirth, C. (1995). A generalized permutation approach to job shop scheduling with genetic algorithms. Operations-Research-Spektrum, 17(2), 87–92. https://doi.org/10.1007/BF01719250



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
from or_benchmark import BenchmarkCollection
import matplotlib.pyplot as plt
from PIL import Image
from typing import List

from individual import * #Individual, QChromosomeRepairPermutationEncoding, QChromosomePositionEncoding, QChromosomeHashPermutationEncoding, QChromosomeHashMultisetEncoding, QChromosomeHashMultisetImprovedEncoding
import datetime

class Node:
    """Class representing each individual node in the linked list object LinkedList
    """
    # The node value stored
    node_value = None
    # The reference to the next node
    next_node = None

    def __init__(self, node_value, next_node):
        """Initialization of the node

        Parameters
        ----------
        node_value : any
            The value stored at the position in the Linked List that this node should represent.
        next_node : Node
            Reference to the next node in the linked list. None if it is the last node in the list.
        """
        self.node_value = node_value
        self.next_node = next_node

    def get_next(self):
        # Returns the reference to the next node/element in the list
        return self.next_node
    
    def get_value(self):
        # Returns the stored value of the node
        return self.node_value
    
    def set_next(self, next_node):
        # Helper method for adding the reference to the next list node
        self.next_node = next_node
    

class LinkedList:
    """Limited implementation of the linked list suited specially for the GOX algorithm used in the classical NSGA-II implementation
    """
    # Reference to the first node in the list
    start_node = None
    # Reference to the last node in the list
    end_node = None
    # The length of the array
    length = 0

    def __init__(self, initial_array: np.ndarray):
        """Initialization of the LinkedList object.

        Parameters
        ----------
        initial_array : np.ndarray
            The array that should be converted to linked list.

        Raises
        ------
        Exception
            The provided array is of an unsupported format.
        """
        if not type(initial_array) == np.ndarray or not len(initial_array.shape) == 1:
            # Raise exception if the provided array is unsupported
            raise Exception("The provided iterable was not recognized.")
        
        if len(initial_array) == 0:
            # If the array should be empty, skip initialization
            return
        
        # Create the last node in the list first
        self.end_node = previous_node = Node(initial_array[-1], None)
        self.length += 1
        for i in reversed(range(len(initial_array)-1)):
            # For each element ot add, create Node referencing the previously created node and store the reference for the next iteration
            previous_node = Node(initial_array[i], previous_node)
            self.length += 1
        # Set the last created node as the start of the list
        self.start_node = previous_node

    def remove_node(self, delete_node: Node, previous_node: Node):
        """Method for removing a node from the linked list.

        Parameters
        ----------
        delete_node : Node
            Reference to node to be deleted.
        previous_node : Node
            Reference to the preceeding node to couple past the delte_node

        Returns
        -------
        any
            The value of the removed node is returned
        """
        temp_value = delete_node.get_value()

        if delete_node == self.start_node:
            # If the node to be deleted is at index 0
            self.start_node = delete_node.get_next()
            return temp_value
        
        if delete_node == self.end_node:
            # If the deleted node is at index -1
            self.end_node = previous_node
            return temp_value
        
        previous_node.set_next(delete_node.get_next())
        self.length -= 1
        return temp_value
    
    def insert_nodes(self, new_values: np.ndarray, insert_after_node: Node):
        """Method for inserting an array of nodes after a given node.

        Parameters
        ----------
        new_values : np.ndarray
            The array containing the values to be added
        insert_after_node : Node
            The node that should preceed the new values

        Raises
        ------
        Exception
            The array provided as argument does not hav ethe correct format
        """
        if not type(new_values) == np.ndarray or not len(new_values.shape) == 1:
            # Check that the inputed array is supported
            raise Exception("The provided iterable was not recognized.")
        
        if not insert_after_node:
            # If the array is inserted in the ront
            cur_next = self.start_node
            for i in reversed(range(len(new_values))):
                # For each new node value, from the last to first (reversed)
                # Create new node with the current next as next node, and store the new node as next reference
                cur_next = Node(new_values[i], cur_next)
                # Increment length
                self.length += 1
            # Set the new start node
            self.start_node = cur_next
        elif insert_after_node == self.end_node:
            # The array is appended to the list
            cur_next = None
            # Set new end node
            self.end_node = cur_next = Node(new_values[-1], cur_next)
            for i in reversed(range(len(new_values)-1)):
                # For each new node value, from the last to first (reversed)
                # Create new node with the current next as next node, and store the new node as next reference
                cur_next = Node(new_values[i], cur_next)
                # Increment length
                self.length += 1

            insert_after_node.set_next(cur_next)
        else:
            # Get reference to the node proceeding the new nodes
            cur_next = insert_after_node.get_next()
            for i in reversed(range(len(new_values))):
                # For each new node value, from the last to first (reversed)
                # Create new node with the current next as next node, and store the new node as next reference
                cur_next = Node(new_values[i], cur_next)
                # Increment length
                self.length += 1

            insert_after_node.set_next(cur_next)



    def to_ndarray(self) -> np.ndarray:
        resulting_array = []
        cur_node = self.start_node
        while not self.end_node == cur_node:
            resulting_array.append(cur_node.get_value())
            cur_node = cur_node.get_next()
        
        resulting_array.append(self.end_node.get_value())
        return np.array(resulting_array)

    
    def __str__(self):
        # Method for implicitly converting the object into a string representation - the class supports print(LinkedListinstance)
        cur_node = self.start_node
        list_string = "["

        while not cur_node is self.end_node:
            list_string += str(cur_node.get_value()) + ", "
            cur_node = cur_node.get_next()

        list_string += str(cur_node.get_value()) + "]"
        return list_string
    
    def __iter__(self):
        # Initialization of the iterator
        self.current_iterator_node = self.start_node
        self.iterator_finished = False
        return self
    
    def __next__(self):
        # Definition of each iteration of the iterator
        if self.iterator_finished:
            raise StopIteration
        
        if self.current_iterator_node == self.end_node:
            self.iterator_finished = True
            return self.end_node
        else:
            temp_node = self.current_iterator_node
            self.current_iterator_node = self.current_iterator_node.get_next()
            return temp_node
        
    def __len__(self):
        # hook method to support len()
        return self.length



class Population:
    # Consists of the parent and ofspring population P_t and Q_t to form the full population R_t
    def __init__(self, 
                 N: int, 
                 decoding_method: str, 
                 n_jobs: int, 
                 n_machines: int, 
                 jssp_problem: np.ndarray,
                 activate_schedule: bool = False,
                 time_log: bool=False,
                 objectives: List[str] = ["max_completion_time", "mean_flow_time"],
                 population_array: np.ndarray = np.array([])
                ):
        
        # Define the total population size
        self.N = N
        self.n_jobs: int = n_jobs
        self.n_machines: int = n_machines
        self.jssp_problem: np.ndarray = jssp_problem
        self.decoding_method: str = decoding_method
        self.activate_schedule: bool = activate_schedule
        self.time_log: bool = time_log
        self.objectives: List[str] = objectives
        self.population_array: np.ndarray = population_array

    def initialize_population(self):
        raise NotImplemented()

    def evaluate_fitness(self):
        raise NotImplemented()
            
    #def fast_non_dominated_sort(self):
        # P -> self.R
    #    new_R = np.empty(len(self.R), dtype=[("individual", object), ("domination_count", int), ("dominated_solutions", object)])
    #    cur_start = 0
    #    for p_i in range(cur_start, len(self.R)):
    #        S_p = []
    #        n_p = 0
    #        Q = []
    #        for q_i = 0

    def non_dominated_sorting(self):
        # Sort the entire population
        self.front_start_index: List = [] # Contains start index for each front
        cur_start = 0
        cur_swap_index = 0
        while cur_start < self.N:
            for i in range(cur_start, len(self.R)):
                # For each individual go through the rest to find the best possible individual (that dominates all solutions)
                dominated = False
                for j in range(cur_start, len(self.R)):
                    # swap if J dominates the
                    cur_comparison = np.array(self.R[j].cur_fitness) <= np.array(self.R[i].cur_fitness)
                    if (i != j) and (cur_comparison).all():
                        # Break the loop if the individual i is dominated -> means that the individual does not belong to the current front
                        if not (np.array(self.R[j].cur_fitness) == np.array(self.R[i].cur_fitness)).all():
                            # There are cases where the solutions are exactly equal. In that case we declare that j does not dominate i
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

        # Add the last front
        if self.front_start_index[-1] < self.N:
            self.front_start_index.append(self.N)


    def get_front_range(self, i: int) -> List:
        """This method returns the indices for the current front i.

        Parameters
        ----------
        i : int

        Returns
        -------
        List
            Returns list with [start_index, stop_index+1, length_of_front]
        """
        if i + 1 >= len(self.front_start_index):
            # Handeling the last front - rest of the population
            return([self.N, 2*self.N, self.N])
        if i + 2 >= len(self.front_start_index):
            # Handeling the second to last front
            cur_length = self.N - self.front_start_index[i]
            return([self.front_start_index[i], self.N, cur_length])
        else:
            cur_length = self.front_start_index[i+1] - self.front_start_index[i]
            return([self.front_start_index[i], self.front_start_index[i+1], cur_length])
        
    def get_performance(self, is_last_iteration=False) -> np.ndarray:
        """This method is used to calculate the performance metrics for the non-dominated front

        Returns
        -------
        np.ndarray
            Array containing the metrics: makespan [min, avg], mean flow time [min, avg], spread
        """
        if is_last_iteration:
            cur_spread, cur_length = self.calculate_spread_euclidian(is_last_iteration)
        else:
            cur_spread, cur_length = self.calculate_spread_euclidian(is_last_iteration)

        cur_range = self.get_front_range(0)
        result = {"Makespan": {"Avg" : 0, "Min": np.inf, "Max" : 0}, 
                  "Mean flow time": {"Avg" : 0, "Min": np.inf, "Max" : 0}, 
                  "Spread" : cur_spread, 
                  "n_fronts" : len(self.front_start_index),
                  "n_non_dominated_solutions" : cur_length
                }
        
        for i in range(cur_range[1]):
            # Find avg makespan
            result["Makespan"]["Avg"] += self.R[i].schedule.max_completion_time
            # Find minimum makespan
            if result["Makespan"]["Min"] > self.R[i].schedule.max_completion_time:
                result["Makespan"]["Min"] = self.R[i].schedule.max_completion_time
            # Find maximum makespan
            if result["Makespan"]["Max"] < self.R[i].schedule.max_completion_time:
                result["Makespan"]["Max"] = self.R[i].schedule.max_completion_time

            # Find avg mean flow time
            result["Mean flow time"]["Avg"] += self.R[i].schedule.mean_flow_time
            # Find minimum mean flow time
            if result["Mean flow time"]["Min"] > self.R[i].schedule.mean_flow_time:
                result["Mean flow time"]["Min"] = self.R[i].schedule.mean_flow_time
            # Find maximum mean flow time
            if result["Mean flow time"]["Max"] < self.R[i].schedule.max_completion_time:
                result["Mean flow time"]["Max"] = self.R[i].schedule.max_completion_time

        result["Makespan"]["Avg"] = result["Makespan"]["Avg"] / cur_range[2]
        result["Mean flow time"]["Avg"] = result["Mean flow time"]["Avg"] / cur_range[2]
        return result

    def calculate_spread_euclidian(self, is_last_iteration=False):
        """This method calculates the spread metric proposed by (Deb et al., 2002)

        uses the crowding_distance_data attribute
            
        Returns
        -------
        float
            The spread value for the current Pareto-front
        """       
        if is_last_iteration:
            # Remove the duplicated solutions
            temp_nd = self.nd_front_crowding_distance
            self.nd_front_crowding_distance = copy.deepcopy(temp_nd)
            counter = 0
            while counter < len(self.nd_front_crowding_distance):
                cur_keep_individual = self.nd_front_crowding_distance[counter]
                k = counter+1
                for indiv in self.nd_front_crowding_distance[counter+1:]:
                    if indiv["makespan"] == cur_keep_individual["makespan"] and indiv["mean flow time"] == cur_keep_individual["mean flow time"]:
                        self.nd_front_crowding_distance = np.delete(self.nd_front_crowding_distance, k)
                    else:
                        k += 1
                counter += 1

        if len(self.nd_front_crowding_distance) < 2:
            # If there is only one solution that is non-dominated, 
            # return infinity to emphazise the need for more than 
            # one solutions in the converged front
            return (np.inf, 1)
        
        #d_extreme = (self.crowding_distance_data["makespan"].max() - self.crowding_distance_data["makespan"].min())**2
        #d_extreme += (self.crowding_distance_data["mean flow time"].max() - self.crowding_distance_data["mean flow time"].min())**2
        
        # First the distance between the extreme solutions in the estimated pareto-front Q and the optimal Pareto-Front
        # Note that the optimal pareto front is not known in this case and the extremes are set to 0 for both objectives.
        # As a result the distance is the minimum value for the metrics. 
        # If d_extreme -> 0, we now that the pareto fron covers the optimal front in bredth
        d_extreme = self.nd_front_crowding_distance["makespan"].min() + self.nd_front_crowding_distance["mean flow time"].min() #np.sqrt(d_extreme)
        # Initializing the total distance value
        di_sum = 0
        # List of distances
        di_list = np.empty(len(self.nd_front_crowding_distance)-1)
        counter = 0
        
        # Now each successive distance between solutions in the non-dominated set is found
        self.nd_front_crowding_distance["makespan"].sort()
        for i in range(1, len(self.nd_front_crowding_distance)):
            # For each solution in the pareto front
            di = (self.nd_front_crowding_distance["makespan"][i] - self.nd_front_crowding_distance["makespan"][i-1])**2
            di += (self.nd_front_crowding_distance["mean flow time"][i] - self.nd_front_crowding_distance["mean flow time"][i-1])**2
            di = np.sqrt(di)
            di_list[i-1] = di
            di_sum += di
            counter += 1

        # Find the mean distance
        d_mean = di_sum/counter
        # Find the deviation from the mean. If the deviaiton -> 0, then the solutions are equally spaced between each other
        di_mean_sum = np.sum(np.abs(di_list - d_mean))
        denominator = (d_extreme + len(di_list)*d_mean)
        # Return the calculated spread
        S = (d_extreme + di_mean_sum)/denominator
        return_length = len(self.nd_front_crowding_distance)
        if is_last_iteration:
            # Set nd back to containing the duplicates to avoid interfering with the update operation
            self.nd_front_crowding_distance = temp_nd

        return (S, return_length)

    def calculate_spread_cd(self):
        # Measure of how diverse the pareto front is.
        # Two metrics: one with crowding distance and one with the euclidian distance between objective functions
        # The crowding distance of the non dominated front is stored in self.crowding_distance_data
        # 1. find the extremal distances
        d_extreme = self.nd_front_crowding_distance["makespan"].max() - self.nd_front_crowding_distance["makespan"].min() 
        d_extreme += self.nd_front_crowding_distance["mean flow time"].max() - self.nd_front_crowding_distance["mean flow time"].min()
        # 2. Find the mean of the distnace values
        d_mean = self.nd_front_crowding_distance["cd"].mean()
        di_sum = np.sum(self.nd_front_crowding_distance["cd"] - d_mean)
        spread = (d_extreme + di_sum)/(d_extreme + len(self.nd_front_crowding_distance["cd"])*d_mean)
        return spread
        

    def crowding_distance_sort_all_fronts(self):
        """This method is used to perform crowding distance sorting on all fronts in the population
        """
        for i in range(len(self.front_start_index)):
            # Sort every front on chrowding distance
            cur_front = self.get_front_range(i)
            if i == 0:
                # Store the first front as attribute
                self.nd_front_crowding_distance = self.crowding_distance_sort_front(cur_front, return_cd_data=True)
            else:
                self.crowding_distance_sort_front(cur_front, return_cd_data=False)

    def swap(self, i, j):
        temp = self.R[i]
        self.R[i] = self.R[j]
        self.R[j] = temp

    def crowding_distance_sort_front_old(self, cur_range, return_cd_data=False):
        # This method is used to perform crowding distance sorting on the last non-dominated front in self.start_stop_fronts.
        # Needs to be executed after non-dominated sorting
        # As a result the last front is sorted and the population is ready for parent selection and recombination
        #cur_range = self.start_stop_fronts[-1]
        range_len = cur_range[2]
        order_list = []
        # Create the order list - contains list of position in R and the objective values
        for i, individual in enumerate(self.R[cur_range[0] : cur_range[1]]):
            # Add the inherent position of the current solution
            cur_order_list = [i]
            # Add the objective values
            cur_order_list.extend([getattr(individual.schedule, objective) for objective in self.objectives])
            # add the cd element to the array
            cur_order_list.append(0)
            # The order list for the current individual is added to front order list
            order_list.append(cur_order_list)

        for i in range(1, len(self.objectives)+1):
            # For each objective, order the individuals in the order list such that they do get the different 
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
        
        if return_cd_data:
            return order_list
        

    def crowding_distance_sort_front(self, cur_range, return_cd_data=False) -> np.ndarray:
        # This method is used to perform crowding distance sorting on the last non-dominated front in self.start_stop_fronts.
        # Needs to be executed after non-dominated sorting
        # As a result the last front is sorted and the population is ready for parent selection and recombination
        #cur_range = self.start_stop_fronts[-1]
        range_len = cur_range[2]
        dtype_list = [("position", int), ("makespan", int), ("mean flow time", float), ("cd", float)]
        order_list = np.empty(range_len,  dtype=dtype_list)
        # Create the order list - contains list of position in R and the objective values
        for i, individual in enumerate(self.R[cur_range[0] : cur_range[1]]):
            # Add the inherent position of the current solution
            order_list[i] = (i, individual.schedule.max_completion_time, individual.schedule.mean_flow_time, 0)
            

        if len(order_list) > 2:
            # Sort on makespan
            order_list.sort(order="makespan")
            # Set boundaries to infinity
            order_list["cd"][0] = np.inf
            order_list["cd"][-1] = np.inf
            # update crowding distance for each individual
            pre_individual_index = np.arange(0, len(order_list)-2)
            post_indvidual_index = np.arange(2, len(order_list))

            cd_denominator = (order_list["makespan"][-1] - order_list["makespan"][0])
            if cd_denominator == 0:
                order_list["cd"][1:-1] = 0
            else:    
                order_list["cd"][1:-1] += (order_list["makespan"][post_indvidual_index] - order_list["makespan"][pre_individual_index])/cd_denominator
            
            # Sort on mean flow time
            order_list.sort(order="mean flow time")
            # Set boundaries to infinity
            order_list["cd"][0] = np.inf
            order_list["cd"][-1] = np.inf
            # update crowding distance for each individual
            pre_individual_index = np.arange(0, len(order_list)-2)
            post_indvidual_index = np.arange(2, len(order_list))

            cd_denominator = (order_list["mean flow time"][-1] - order_list["mean flow time"][0])
            if cd_denominator == 0:
                # If all solutions are similar in fitness the cd denominator is 0. This is handled by setting all values to infinity
                order_list["cd"][1:-1] = 0
            else:    
                order_list["cd"][1:-1] += (order_list["mean flow time"][post_indvidual_index] - order_list["mean flow time"][pre_individual_index])/cd_denominator
        else:
            order_list["cd"][0] = np.inf
            order_list["cd"][-1] = np.inf

        # Sort on distance
        order_list.sort(order="cd")
        # Reverse the sorting since crowding distance is maximized
        order_list = np.flip(order_list, axis=0)
        # Reflect the new order on the population    
        self.R[cur_range[0] : cur_range[1]] = self.R[cur_range[0] + order_list["position"]]
        
        if return_cd_data:
            return order_list
        
    
    def plot_fronts(self):
        # This method produces a scatter plot for the different fronts
        
        #color_list = np.ones((len(self.R), 4))
        point_list = np.zeros((len(self.R), 2))

        for i in range(len(self.front_start_index)):
            cur_range = self.get_front_range(i)
            for j in range(cur_range[0], cur_range[1], 1):
                point_list[j, :] = self.R[j].cur_fitness
                #color_list[j, :] = plt.cm.gist_rainbow(i/len(self.front_start_index)) # viridis
                
        return point_list #[point_list, color_list]


class ClassicalPopulation(Population):

    def __init__(self,
                 N: int, 
                 decoding_method: str,
                 n_jobs,
                 n_machines,
                 jssp_problem,
                 activate_schedule,
                 tournament_size: int,
                 mating_pool_size: int,
                 mutation_probability: float

            ):
        super().__init__(N, decoding_method, n_jobs, n_machines, jssp_problem, activate_schedule)
        self.tournament_size = tournament_size
        self.mating_pool_size = mating_pool_size
        self.mutation_probability = mutation_probability

        if len(self.population_array) == 0:
            self.initialize_population()
        else:
            self.R = self.population_array

    def initialize_population(self):
        # P is index 0 - N-1, while Q is index N - 2N
        self.R = np.empty(2*self.N, dtype=Individual)
        for i in range(len(self.R)):
            self.R[i] = PermutationChromosome.create_permutation_chromosome(self.n_jobs, self.n_machines)


    def evaluate_fitness(self):
        cur_chromosome: Individual
        for cur_chromosome in self.R:
            cur_chromosome.create_schedule(
                self.decoding_method,
                self.jssp_problem,
                self.activate_schedule
            )
        # Fintess values are available as self.R[i].schedule.max_completion_time

    def select_parents(self):
        """This method performs k-tournament selection to obtain all parents needed for the mating pool.
        The result of this method, is a mating_pool ready for recombination.
        The tournament selection uses the rank of the individuals in the population when determining the winner of the tournament.
        Note that there is replacement, meaning that the same solution can be picked several times.
        Parents are only picked amon the N first individuals of the populaiton.
        """
        self.mating_pool = np.empty(self.mating_pool_size, dtype=Individual)
        for selected_index in range(len(self.mating_pool)):
            # Find the smallest index among the integers representing the index of the array R sorted by non-dominated sorting and then crowding distance
            cur_index = np.min(np.random.randint(0, self.N, self.tournament_size))
            # Adding solution to the mating pool
            self.mating_pool[selected_index] = self.R[cur_index]

        
    def execute_recombination(self):
        # performs crossover on the mating_pool to fill the second half of the population
        offspring_offset = self.N
        offspring_index = 0
        while offspring_index < self.N:
            for m_index in range(0, len(self.mating_pool), 2):
                # For each two pairs in the mating pool determine the donator and the receiver
                donator = self.mating_pool[m_index].permutation
                receiver = self.mating_pool[m_index + 1].permutation
                # Perform the recombination and replace the existing solutions in the population with the offspring produced
                self.R[offspring_offset + offspring_index] = self.gox_crossover(donator, receiver)
                # perform mutation on offspring
                needle = np.random.uniform()
                if self.mutation_probability >= needle:
                    self.R[offspring_offset + offspring_index].permutation = self.job_pair_mutation(self.R[offspring_offset + offspring_index].permutation)
                
                offspring_index += 1
                if offspring_index == self.N:
                    return

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

    def gox_crossover(self, donator: np.ndarray, receiver: np.ndarray):
        """This method performs the Generalized Order Crossover described in (Bierwirth, 1995).

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
        lenght_of_range = int(section*len(donator))
        stop_range = int(start_range + lenght_of_range) % len(donator)
        # Changes are made to the receiver. It is therefore copied here to avoid sideeffects for the method
        linked_receiver = LinkedList(receiver)
        # The child individual is created by modifying the receiver genotype 
        # First find all the genes that correspond to the genes in the donator
        # They are then removed by setting them to -1
        insert_after_node = None
        for j in range(self.n_jobs):
            # For each job - iterate over the receiver 
            donator_counter = 0
            receiver_counter = 0
            previous_node = None
            cur_node = linked_receiver.start_node
            for k in range(len(donator)):
                # For each job_index in m1 check for match in receiver
                if donator[k] == j:
                    donator_counter += 1
                    # If the current job in the donator corresponds to the current job - find the
                    while not previous_node == linked_receiver.end_node:
                        if cur_node.get_value() == j:
                            receiver_counter += 1
                            # Found the matching job number
                            if ((start_range < stop_range) and (k >= start_range and k < stop_range) and donator_counter == receiver_counter) \
                                or ((start_range > stop_range) and not (k < start_range and k >= stop_range) and donator_counter == receiver_counter):
                                # The match is in range
                                if k == start_range:
                                    insert_after_node = previous_node

                                temp_next_node = cur_node.get_next()
                                if insert_after_node == cur_node:
                                    insert_after_node = previous_node
                                    
                                linked_receiver.remove_node(cur_node, previous_node)
                                cur_node = temp_next_node
                                break
                            else:
                                previous_node = cur_node
                                cur_node = cur_node.get_next()
                                break

                        previous_node = cur_node
                        cur_node = cur_node.get_next()

        # Find the injection position
        #injection_index = np.where(copy_receiver == -1)[0][0]
        #child_chromosome = np.empty(len(copy_receiver), dtype=int)
        # Need to distinguish between wrapper and normal case
        if start_range > stop_range:
            # Wrapper case
            # Insert the first elements of the crossover string at the back
            linked_receiver.insert_nodes(donator[start_range:], insert_after_node)
            # Insert the last elements of the crossover string at the beginning
            linked_receiver.insert_nodes(donator[:stop_range], None)
        else:
            # Normal case
            # Inject crossover string
            linked_receiver.insert_nodes(donator[start_range:stop_range], insert_after_node)

        child_chromosome = linked_receiver.to_ndarray()
        return PermutationChromosome(child_chromosome, self.n_jobs, self.n_machines)

class QMEAPopulation(Population):
    def __init__(self,
                 N: int,
                 reset_fraction: float,
                 decoding_method: str,
                 n_jobs: int,
                 n_machines: int,
                 jssp_problem: np.ndarray,
                 individual_cfg: DictConfig,
                 rotation_angles: str, 
                 group_partitions: int,
                 activate_schedule: bool=False,
                 time_log: bool=False,
                 individual_type="QChromosomeBaseEncoding"
        ):
        super().__init__(N, decoding_method, n_jobs, n_machines, jssp_problem, activate_schedule, time_log)
        self.individual_type = individual_type
        self.reset_fraction = reset_fraction
        self.rotation_angles = rotation_angles
        self.Individual_cfg = individual_cfg
        self.group_partitions = group_partitions
        self.initialize_population(time_log)
        self.evaluate_fitness()

    def evaluate_fitness(self):
        for cur_chromosome in self.R:
            # make measurement
            cur_chromosome.measure()
            # Convert bit string to operation based representation
            cur_chromosome.convert_permutation()
            # Create schedule to evaluate fitness
            cur_chromosome.create_schedule(
                self.decoding_method, 
                self.jssp_problem,
                self.activate_schedule
            )
        # Fintess values are available as self.R[i].schedule.max_completion_time

    def initialize_population(self, time_log: bool=False):
        """Method for populating the population with individuals. Generates random qubit chromosomes.

        Parameters
        ----------
        time_log : bool, optional
            Logging durations of different components in representation, by default False
        """
        # P is index 0 - N-1, while Q is index N - 2N
        cur_individual_type = eval(self.individual_type)
        self.R = np.empty(2*self.N, dtype=cur_individual_type)
        for i in range(len(self.R)):
            self.R[i] = cur_individual_type(self.n_jobs, self.n_machines, self.Individual_cfg, time_log=time_log)

    def execute_quantum_update(self, c: int, c_tot: int):
        """This method is used to perform recombination for the QMEA algorithm
            
        Parameters
        ----------
        c : int
            The current generation number
        c_tot : int
            The total generation to run

        Raises
        ------
        Exception
            The groups of solutions should contains more solutions than 0.
        """
        population_size = self.N*2*(1-self.reset_fraction)
        # Divide the parents into equal groups
        S = int(np.floor(population_size/self.group_partitions))
        if S <= 0:
            raise Exception("The groups of solutions should contains more solutions than 0. Please adjust the n_groups parameter.")
        
        group = np.arange(1, self.group_partitions, 1)*S
        for g in group:
            for s in range(S):
                # For each solution in the best group, use it to rotate the solution
                # Solutions in other groups are given by S*group + s
                self.R[g + s].rotate(self.R[s], c=c, c_tot=c_tot, n_groups=self.group_partitions, cur_group=g, rotation_angles=self.rotation_angles)

        # Reset remaining solutions
        reset_index = int(self.N*2*(1-self.reset_fraction))
        for remaining in self.R[reset_index:]:
            remaining.reset_chromosome()

        # Set the amplitudes of the best group
        for s in range(S):
            self.R[s].binary_chromosome[0, ...] = np.logical_not(self.R[s].x)
            self.R[s].binary_chromosome[1, ...] = self.R[s].x
            # Remove mutations from the best group.
            self.R[s].mutation_flags *= 0

        # Reset individuals that are duplicated
        cur_range = self.get_front_range(0)
        front_indexes = np.arange(cur_range[0], cur_range[1]).tolist()
        while len(front_indexes) > 0:
            cur_makespan, cur_flow = self.R[front_indexes[0]].cur_fitness
            temp_front_indexes = []
            for j in range(1, len(front_indexes)):
                comp_makespan, comp_flow = self.R[front_indexes[j]].cur_fitness
                if cur_makespan == comp_makespan and cur_flow == comp_flow:
                    # Set amplitudes to 1/sqrt(2)
                    self.R[front_indexes[j]].binary_chromosome[0, :] = np.sqrt(2)**(-1)
                    self.R[front_indexes[j]].binary_chromosome[1, :] = np.sqrt(2)**(-1)
                    temp_front_indexes.append(j)
                    

            front_indexes = temp_front_indexes