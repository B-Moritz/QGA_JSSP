import numpy as np
import pandas as pd
from typing import List

from operations import Operation

def create_random_jssp_problem(n_jobs, n_machines):
    """This method is used to create random jssp problems in the similar format as the or library benchmarks (see or_benchmark.py)"""
    max_min_duration = np.random.randint(low=1, high=20, size=1)
    problem_matrix = np.random.randint(low=max_min_duration, high=max_min_duration + 10, size=(2, n_jobs, n_machines))
    for j in range(n_jobs):
        # Arrange each machine sequence such that it is a permutation of numbers between 0 and n_machines
        problem_matrix[0][j][:] = np.random.permutation(n_machines)

    return problem_matrix


def create_m_rep_permutation(n_jobs: int, n_machines: int) -> np.ndarray:
    """Method used to generate a random permutation with repetition where the number of repetition per job int is given by n_machines."""
    # create a random permutation of size j*m. 
    indexes = np.random.permutation(n_machines*n_jobs)
    # The empty chromosome
    j_rep_permutation = np.empty_like(indexes)

    for i in range(len(indexes)):
        # The integers are used as indexes for plasing the job number between 0 and j in the chromosome array
        cur_index = indexes[i]
        j_rep_permutation[cur_index] = i % n_jobs

    # The chromosome contains a permutation with repetition and is returned
    return j_rep_permutation

def apply_operation_based_bierwirth(
        n_jobs: int, 
        n_machines: int, 
        j_rep_permutation: np.ndarray,
        jssp_problem: np.ndarray,
    ) -> List[Operation]:

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
        # Find the next machine in the technological sequence
        cur_machine = jssp_problem[0][cur_job][T_counter[cur_job]]
        # First determine which machine the job should be on
        if m_start_t[cur_machine] >= j_start_t[cur_job]:
            cur_start = m_start_t[cur_machine]
        else:
            cur_start = j_start_t[cur_job]

        cur_duration = jssp_problem[1][cur_job][T_counter[cur_job]]
        m_start_t[cur_machine] = cur_start + cur_duration
        j_start_t[cur_job] = cur_start + cur_duration

        operation_list[k] = Operation(cur_job, cur_machine, cur_duration, cur_start)
        # Increment the mahcine counter
        T_counter[cur_job] += 1

    return operation_list