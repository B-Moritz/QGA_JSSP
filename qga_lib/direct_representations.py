"""Utility functions for working with the representation of jssp.

References:
* Bierwirth, C. (1995). A generalized permutation approach to job shop scheduling with genetic algorithms. Operations-Research-Spektrum, 17(2), 87–92. https://doi.org/10.1007/BF01719250


Returns
-------
_type_
    _description_
"""

import numpy as np
import pandas as pd
from typing import List
import pdb

from operations import Operation

def create_random_jssp_problem(n_jobs, n_machines):
    """This method is used to create random jssp problems in the similar format as the or library benchmarks (see or_benchmark.py)"""
    max_min_duration = np.random.randint(low=1, high=20, size=1)
    problem_matrix = np.random.randint(low=max_min_duration, high=max_min_duration + 10, size=(2, n_jobs, n_machines))
    for j in range(n_jobs):
        # Arrange each machine sequence such that it is a permutation of numbers between 0 and n_machines
        problem_matrix[0][j][:] = np.random.permutation(n_machines)

    return problem_matrix

def create_multiset_permutation(j, m):
    """This method is used to create a random permutation of multisets.
       The variable j is sthe number of different objects and m is the number of identical objects

    Parameters
    ----------
    j : int
        Number of jobs in the permutation (number of unique objects).
    m : int
        Number of repetitions of the job number in the permutation (number of machines a job can be assigned to).

    Returns
    -------
    _type_
        _description_
    """
    # Create the first permutation of multiset (starts at job 0 to job number j-1)
    start_multiset = np.repeat(np.arange(j), m)
    # Return a random shuffle of the initial permutaiton
    return np.random.permutation(start_multiset)

#def create_m_rep_permutation(n_jobs: int, n_machines: int) -> np.ndarray:
#    """Method used to generate a random permutation with repetition where the number of repetition per job int is given by n_machines."""
#    # create a random permutation of size j*m. 
#    indexes = np.random.permutation(n_machines*n_jobs)
#    # The empty chromosome
#    j_rep_permutation = np.empty_like(indexes)
#
#   for i in range(len(indexes)):
#        # The integers are used as indexes for placing the job number between 0 and j in the chromosome array
#        cur_index = indexes[i]
#        j_rep_permutation[cur_index] = i % n_jobs
#
#    # The chromosome contains a permutation with repetition and is returned
#    return j_rep_permutation


def apply_operation_based_bierwirth(
        n_jobs: int, 
        n_machines: int, 
        j_rep_permutation: np.ndarray,
        jssp_problem: np.ndarray,
    ) -> List[Operation]:
    """This function produces a schedule from an operation based representation (permutation of multisets) after the 
    method precented by (Bierwirth, 1995).

    Parameters
    ----------
    n_jobs : int
        Number of jobs to schedule
    n_machines : int
        Number of machines to schedule the jobs to
    j_rep_permutation : np.ndarray
        The permutation representation of the schedule to be converted
    jssp_problem : np.ndarray
        A three dimensional array containing the problem that is to be optimized.
        shape (2, j, m) - The first axis separates job numbers at index 0 and processing time at index 1

    Returns
    -------
    List[Operation]
        _description_
    """

    # The finished schedule is stored in the following list
    operation_list = np.empty(n_machines*n_jobs, dtype=Operation)
    # The machine and job time counters used to determine the start time for the next operations
    m_start_t = np.zeros(n_machines)
    j_start_t = np.zeros(n_jobs)
    # machine sequence counter to determine which machine is next for each job
    T_counter = np.zeros(n_jobs, dtype=int)

    for k in range(len(j_rep_permutation)):
        # For each job number in the permutation, schedule it at the first possible time
        cur_job = j_rep_permutation[k]
        try:
            # Find the next machine in the technological sequence
            cur_machine = jssp_problem[0][cur_job][T_counter[cur_job]]
        except:
            print(j_rep_permutation)
            print(np.bincount(j_rep_permutation))
            exit()
        # First determine the start time for the operation
        if m_start_t[cur_machine] >= j_start_t[cur_job]:
            # Comapre the next possible start time of the machine and the job, 
            # and select the highest as the current starting time for the operation
            cur_start = m_start_t[cur_machine]
        else:
            cur_start = j_start_t[cur_job]

        # Extract the current durration from jssp problem definiton
        cur_duration = jssp_problem[1][cur_job][T_counter[cur_job]]
        # Set the new start times for the machine and for the job
        m_start_t[cur_machine] = cur_start + cur_duration
        j_start_t[cur_job] = cur_start + cur_duration
        # Create the operation and add it to the operation list
        operation_list[k] = Operation(cur_job, cur_machine, cur_duration, cur_start)
        # Increment the machine counter for the current job to allow progress
        T_counter[cur_job] += 1

    return operation_list