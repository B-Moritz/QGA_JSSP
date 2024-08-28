# This file contains the object representing an individual/chromosome
import numpy as np

from operations import Operation
from schedules import Schedule
from direct_representations import *

from typing import List

class Individual:

    def __init__(self, permutation: np.ndarray, n_jobs: int, n_machines: int):
        
        if len(permutation) != n_jobs*n_machines:
            raise Exception("Bad arguments for number of jobs and machines")
        
        self.genotype: np.ndarray = permutation
        self.n_machines: int = n_machines
        self.n_jobs: int = n_jobs

    def create_individual(n_jobs: int, n_machines: int):
        # Generator funciton for creating an individual object
        permutation = create_m_rep_permutation(n_jobs, n_machines)
        return Individual(permutation)
    
    def create_schedule(self, 
                        method: str, 
                        jssp_problem: np.ndarray,
                        objectives: List[str],
                        activate_schedule=False,
                    ):
        try:
            encoding_method = eval(method)
        except NameError as e:
            raise NameError(f"The provided identifier for chromosome encoding was not recognized. Provided identifier: {method}")

        operation_list = encoding_method(
            self.n_jobs, 
            self.n_machines,
            self.genotype,
            jssp_problem
        )

        self.schedule = Schedule(operation_list, self.n_jobs, self.n_machines, jssp_problem)
        if activate_schedule:
            self.schedule.activate_schedule()

        # Make fitness evaluations
        self.cur_fitness = np.array([self.schedule.eval(f"get_{objectives[0]}")(), self.schedule.eval(f"get_{objectives[1]}")()])
        #self.schedule.get_flow_sum()
        #self.schedule.get_mean_flow_time()
    
    