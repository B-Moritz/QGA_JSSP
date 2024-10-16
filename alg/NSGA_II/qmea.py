# This file contains the QMEA implementation for the jssp problem
import pdb

from base_nsga_ii import JsspAlgorithm
from nsga_population import QMEAPopulation
from omegaconf import DictConfig

import numpy as np
            
class QMEA(JsspAlgorithm):

    def __init__(self,
                 n_iterations, 
                 pop_object: QMEAPopulation,
                 activate_logging=False,
                ):
        super().__init__(n_iterations, activate_logging)
        # Initialize the population - This populates the population with randomly generated individuals
        self.pop_object = pop_object


    def execute(self):
        """This generator function constitutes the body of the algorithm main loop for the QMEA.
        """
        # sort the population
        self.pop_object.non_dominated_sorting()
        self.pop_object.crowding_distance_sort_all_fronts()
        self.pop_object.execute_quantum_update()
        
        cur_result = self.pop_object.get_performance()
        cur_result["Iteration"] = 1
        self.print_performance(cur_result)

        if self.activate_logging:
            yield(cur_result)

        for iteration_number in range(2, self.n_iterations):
            self.pop_object.evaluate_fitness()
            self.pop_object.non_dominated_sorting()
            self.pop_object.crowding_distance_sort_all_fronts()
            self.pop_object.execute_quantum_update()
            
            cur_result = self.pop_object.get_performance()
            cur_result["Iteration"] = iteration_number
            self.print_performance(cur_result)
            if self.activate_logging:
                yield(cur_result)