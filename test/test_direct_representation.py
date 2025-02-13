"""Test code for representation library"""

# Libraries used to prepare tests
import unittest
import numpy as np
# Libraries to test
from direct_representations import create_m_rep_permutation
from direct_representations import apply_operation_based_bierwirth
from direct_representations import create_random_jssp_problem

class test_permutation_representation(unittest.TestCase):

    def test_permutation_generation(self):
        """This method is used to test the generation of permutations with repetitions."""
        # Test cases (n_jobs, n_machines)
        test_cases = [(3, 3), (4, 4), (5, 5), (3, 5), (10, 10), (15, 10)]

        for n_jobs, n_machines in test_cases:
            #print(f"Test case: {n_jobs}x{n_machines}")
            # Testing each test case
            base_msg = f"Error with {n_jobs}x{n_machines} test case."
            # Expecting permutation of length 9, integers between 0 and 3 and the n_machines number of repetitions
            res = create_m_rep_permutation(n_jobs, n_machines)
            self.assertEqual(
                len(res), 
                n_jobs*n_machines, 
                msg=base_msg + f"Unexpected length of permutation."
            )

            count_inspection = np.unique(res, return_counts=True)
            #print(count_inspection)
            # Check that the values from 0 to n_jobs is in array
            self.assertEqual(
                np.sum(count_inspection[0] - np.arange(0, n_jobs)), 
                0, 
                msg=base_msg + f"unexpected integer value encountered. Job numbers found counter: {count_inspection[0]}"
            )
            # Check that there are correct amount of repetitions
            self.assertEqual(
                np.sum(count_inspection[1] - np.ones(n_jobs)*n_machines), 
                0, 
                msg=base_msg + f"Unexpected number of repetitions. Repetition counter: {count_inspection[1]}"
            )
    
    def test_create_random_jssp_problem(self):
        n_jobs = 4
        n_machines = 5
        problem_matrix = create_random_jssp_problem(n_jobs, n_machines)
        self.assertEqual(problem_matrix.shape[0], 2, msg="Error with shape. Missing a problem matrix")
        self.assertEqual(problem_matrix.shape[1], n_jobs, msg="Missing jobs")
        self.assertEqual(problem_matrix.shape[2], n_machines, msg="Missing machines")
        # Check that the machine sequences are permutations
        for j in range(n_jobs):
            self.assertEqual(np.sum(np.sort(problem_matrix[0][j]) - np.arange(n_machines)), 0, msg="Machine sequence contains repetitions")


    def test_operation_based_bierwirth(self):
        # Test cases (n_jobs, n_machines)
        test_cases = [(3, 3), (4, 4), (5, 5), (3, 5), (10, 10), (15, 10)]

        for n_jobs, n_machines in test_cases:
            base_msg = f"Error with {n_jobs}x{n_machines} test case."
            # Create a schedule
            res = create_m_rep_permutation(n_jobs, n_machines)
            cur_jssp_problem = create_random_jssp_problem(n_jobs, n_machines)
            schedule = apply_operation_based_bierwirth(n_jobs, n_machines, res, cur_jssp_problem)

            # disjunctive constraint - Check that a job is not processed on different machines in parallel
            for i in range(n_jobs):
                #Iterate over the operation list
                cur_start = 0
                disjunctive_constraint_holds = True
                for o in schedule:
                    if o.job == i:
                        if cur_start <= o.start:
                            cur_start = o.start + o.duration
                        else:
                            disjunctive_constraint_holds = False
                            
                self.assertTrue(disjunctive_constraint_holds, msg=f"{base_msg}. Disjunctive constraint does not hold for schedule generated from {res}.")
            
if __name__=="__main__":
    unittest.main()