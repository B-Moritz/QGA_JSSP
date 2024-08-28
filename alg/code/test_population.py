"""Test code for the population class"""

# Libraries used to prepare tests
import unittest
from unittest.mock import Mock
import numpy as np
from individual import Individual
from schedules import Schedule
from base_nsga_ii import Population
import pdb

class test_permutation_representation(unittest.TestCase):

    def test_non_dominated_sorting(self):
        N = 5
        R = np.empty(2*N, dtype=Individual)
        test_case = [[7, 6], [7, 10], [3, 1], [6, 10], [2, 1], [4, 11], [7, 8], [9, 1], [3, 4], [7, 15]]
        
        expected_fronts = [[0, 0], [1, 1], [2, 3], [4, 6]]
        expected_order = [[2, 1], [3, 1], [9, 1], [3, 4], [7, 6], [4, 11], [6, 10], [7, 8], [7, 10],  [7, 15]]
        
        for i in range(len(R)):
            R[i] = Mock(Individual)
            R[i].cur_fitness = test_case[i]

        test_p = Population(N, "apply_operation_based_bierwirth", 3, 3, np.array([]), population_array=R)
        test_p.non_dominated_sorting()

        #for i in test_p.R:
        #    print(i.cur_fitness)

        self.assertEqual(len(test_p.start_stop_fronts), len(expected_fronts))
        self.assertEqual(test_p.start_stop_fronts, expected_fronts)
        
    def test_crowding_distance(self):
        N = 3
        R = np.empty(2*N, dtype=Individual)
        test_case = [[5, 10], [4, 20], [8, 3], [10, 1], [9, 2], [7, 4]]
        
        expected_fronts = [[0, 6]]
        expected_order = [[10, 1], [4, 20], [5, 10], [7, 4], [9, 2], [8, 3]]
        
        for i in range(len(R)):
            R[i] = Mock(Individual)
            R[i].cur_fitness = test_case[i]
            R[i].schedule = Mock(Schedule)
            R[i].schedule.makespan = test_case[i][0]
            R[i].schedule.mean_flow_time = test_case[i][1]

        test_p = Population(N, "apply_operation_based_bierwirth", 3, 3, np.array([]), population_array=R)
        #test_p.non_dominated_sorting()
        test_p.start_stop_fronts = expected_fronts
        test_p.crowding_distance(expected_fronts[0])
        for i, indiv in enumerate(test_p.R[test_p.start_stop_fronts[-1][0] : test_p.start_stop_fronts[-1][1] + 1]):
            self.assertEquals(indiv.cur_fitness, expected_order[i])

        

if __name__=="__main__":
    unittest.main()