# This file contains the object representing an individual/chromosome
import numpy as np
import functools
import time
import math
import decimal
from operations import Operation
from schedules import Schedule
from direct_representations import *
from omegaconf import DictConfig
from sympy.combinatorics.graycode import bin_to_gray, gray_to_bin
import scipy

from typing import List

import pdb
import copy

# Decorator function for measuring the execution time used by the method in question. 
# Results are added in the timer_result attribute of the class
def measure_runtime(method_name: str):
    def decorator_measure_runtime(func):
        functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[0], Individual):
                raise Exception("Please make sure that methods decorated with @measure_runtime is caled with self!")
            
            if args[0].time_log:
                # Start timer
                start_time = time.time()
                # Actual funciton call
                temp_return = func(*args, **kwargs)
                if method_name in args[0].timer_result.keys():
                    args[0].timer_result[method_name] += time.time() - start_time
                else:
                    args[0].timer_result[method_name] = time.time() - start_time
            else:
                temp_return = func(*args, **kwargs)

            return temp_return
        return wrapper
    return decorator_measure_runtime



class Individual:
    watch_individual = False

    def __init__(self, n_jobs: int, n_machines: int, objectives: list, time_log: bool=False, watch_individual: bool=False):
        self.watch_individual = watch_individual
        # Variables for timer components
        self.time_log = time_log
        # This dict contains the measurment for the different parts
        self.timer_result = {}

        self.permutation = np.array([])
        self.n_machines: int = n_machines
        self.n_jobs: int = n_jobs
        self.objectives = objectives
        
    
    @measure_runtime("Create shedule")
    def create_schedule(self, 
                        method: str, 
                        jssp_problem: np.ndarray,
                        #job_permutation: np.ndarray,
                        activate_schedule=False,
                    ):
        """This method acts as schedule builder for the individual. 
        It decodes the n-repetition permutation to a feasable schedule. 
        If activate_schedule is specidied, then the schedules produces are active schedules

        Parameters
        ----------
        method : str
            decoding method
        jssp_problem : np.ndarray
            The technical sequence and durrations of the operations
        activate_schedule : bool, optional
            Flag for producing only active schedules, by default False

        Raises
        ------
        NameError
            If the decoding method was not recognized.
        """
        try:
            encoding_method = eval(method)
        except NameError as e:
            raise NameError(f"The provided identifier for chromosome encoding was not recognized. Provided identifier: {method}")

        if len(self.permutation) == 0:
            # Make sure a job permutation is available
            raise Exception("Please create the n-repetition permutation before attempting to create the schedule.")
        
        # Create operation list
        operation_list = encoding_method(
            self.n_jobs, 
            self.n_machines,
            self.permutation,
            jssp_problem
        )
        # Create schedule
        self.schedule = Schedule(operation_list, self.n_jobs, self.n_machines, jssp_problem, objective_1=self.objectives[0], objective_2=self.objectives[1])
        # Activate schedule
        if activate_schedule:
            self.schedule.activate_schedule()

        # Make fitness evaluations
        self.cur_fitness = np.array([self.schedule.objective_1(), self.schedule.objective_2()])

    def __setattr__(self, name, value):
        if self.watch_individual:
            print(f"Attempting to set {name} to {value}")

        super().__setattr__(name, value)


class PermutationChromosome(Individual):

    def __init__(self, permutation: np.ndarray, n_jobs: int, n_machines: int, objectives: list, time_log: bool=False, watch_individual: bool=False):
        super().__init__(n_jobs, n_machines, objectives, time_log, watch_individual=watch_individual)
        
        if len(permutation) != self.n_jobs*self.n_machines:
            raise Exception("Bad arguments for number of jobs and machines")

        self.permutation: np.ndarray = permutation

    def create_permutation_chromosome(n_jobs: int, n_machines: int, objectives: list, time_log: bool=False, watch_individual: bool=False):
        # Generator funciton for creating an individual object
        permutation = create_multiset_permutation(n_jobs, n_machines)
        return PermutationChromosome(permutation=permutation, n_jobs=n_jobs, n_machines=n_machines, objectives=objectives, time_log=time_log, watch_individual=watch_individual)

class QChromosome(Individual):

    def __init__(self, n_jobs : int, n_machines : int, objectives: list, time_log: bool=False) -> None:
        super().__init__(n_jobs, n_machines, objectives, time_log)
        #conversion_dict = {
        #    "Old" : self.convert_permutation, 
        #    "New" : self.convert_permutation_new, 
        #    "Hashing" : self.convert_permutation_hashing}

    @measure_runtime("Measure qubits")
    def measure(self) -> np.ndarray:
        """This method executes the quantum measurement resulting in a bit string.

        Returns
        -------
        pd.ndarray
            The measured bit string. Shape: n_machines, n_bits
        """
        cur_needles = np.random.uniform(0, 1, self.binary_chromosome.shape[1:])
        self.x = (self.binary_chromosome[0, :, :]**2 < cur_needles).astype(bool) # Note that x adopts the shape of the binary_chromosome except the amplitudes: machine numbers, n_bits
        # Add the static mutations
        self.x = np.abs((self.x - self.mutation_flags))
        return self.x
    
    def convert_bin_to_decimal(self, bin_array: np.ndarray) -> np.ndarray:
        """This method is used to convert the bit string to an integer array

        Parameters
        ----------
        bin_array : np.array
            Array  of boolean values representing the bit string

        Returns
        -------
        np.ndarray
            The resulting integer array
        """
        exponents = np.arange(-bin_array.shape[1]+1, 1).reshape(-1, 1)
        bin_expo = np.repeat(exponents, bin_array.shape[0], axis=1).T*-1 #(np.arange(-len(bin_array)+1, 1)*-1)
        result: np.ndarray = np.sum(bin_array * 2**bin_expo, axis=1)
        return result
    

    def rotate(self, 
               b: object, 
               c: int, 
               c_tot: int,
               n_groups: int,
               cur_group,
               rotation_angles: str="[0.2*np.pi, 0, 0.5*np.pi, 0, 0.5*np.pi, 0, 0.2*np.pi, 0]"
               ):
        """This method executes the rotation operation to perturb the amplitudes of the qubits constituting the quantum chromosome

        Parameters
        ----------
        b : object
            The solution to compare to
        rotation_angles : str, optional
            The rotation angles for each combination of b and x bit, by default "[0.2*np.pi, 0, 0.5*np.pi, 0, 0.5*np.pi, 0.5*np.pi, 0, 0.2*np.pi]"

        Raises
        ------
        ValueError
            If the rotation angle does not contain 8 values.
        """
        cur_b = b.x
        cur_x = self.x
        cached_shape = b.x.shape
        # Test that the rotation angles are valid
        raw_rotation_angles = eval(rotation_angles)
        if len(raw_rotation_angles) != 8:
            raise ValueError("Please specify rotation angle array of length 8.")

        rotation_angles = np.array(raw_rotation_angles) #np.repeat([raw_rotation_angles], self.n_bits*self.n_jobs*self.n_machines, axis=0)
        #rotation_angles = rotation_angles.reshape(self.n_machines, self.n_bits*self.n_jobs, -1)

        signs = np.array([-1, 0, 1, 0, -1, 0, 1, 0]) #np.repeat([np.array([-1, 0, 1, 0, -1, 0, 1, 0])], self.n_bits*self.n_jobs*self.n_machines, axis=0)
        #signs = signs.reshape(self.n_machines, self.n_jobs*self.n_bits, -1)
        #for i in range(len(cur_x)):
        pi = cur_x.astype(int) * (2**2)
        bi = cur_b.astype(int) * (2**1)
        # the best individual has always a better fintess in  this case
        better = int(False) 
        index = (pi + bi + better).ravel() #int(str(pi) + str(bi) + str(better), 2)
        cur_sign = (self.binary_chromosome[0, ...] * self.binary_chromosome[1, ...]) < 0
        cur_angle = rotation_angles[index] * signs[index] * (((-2)*cur_sign.astype(int).ravel())+1)
        cur_angle = cur_angle.reshape(cached_shape)
        # Apply the rotation
        new_a = self.binary_chromosome[0, ...]*np.cos(cur_angle) - self.binary_chromosome[1, ...]*np.sin(cur_angle)
        new_b = self.binary_chromosome[0, ...]*np.sin(cur_angle) + self.binary_chromosome[1, ...]*np.cos(cur_angle)

        self.binary_chromosome[0, ...] = new_a
        self.binary_chromosome[1, ...] = new_b

    
    def reset_chromosome(self):
        # Resets the chromosome such that both amplitudes are equal for each qubit
        self.binary_chromosome = np.ones(self.chromo_shape) * np.sqrt(2)**(-1)
        

class QChromosomeRepairPermutationEncoding(QChromosome):
    
    def __init__(self, n_jobs: int, n_machines: int, individual_cfg: DictConfig, objectives: list, time_log: bool):
        super().__init__(n_jobs, n_machines, objectives, time_log)
        # Determine how many bits are needed to represent the job number
        self.n_bits = int(np.log2(self.n_jobs-1) + 1)
        # Create the amplitudes for the chromosome
        self.length = self.n_bits*self.n_jobs
        self.chromo_shape = (2, self.n_machines, self.n_bits*self.n_jobs)
        self.binary_chromosome = np.ones(self.chromo_shape) # Dimensions: the number of amplitudes, machine number, number of bits for one job sequence
        # Set amplitudes to super position
        self.binary_chromosome = self.binary_chromosome * np.sqrt(2)**(-1)

        # Initialize mutation array
        self.mutation_flags = (np.random.uniform(0, 1, size=(self.n_machines, self.n_bits*self.n_jobs)) < individual_cfg.mutation_rate).astype(int)
        

        # Create P(t) by first performing measurment and then convert permutations
        self.measure()
        self.convert_permutation()

    @measure_runtime("Decode bit string")
    def convert_permutation(self):
        self.permutation = np.zeros((self.n_machines, self.n_jobs), dtype=int)
        cur_val = ""

        # Convert from binary to decimal
        perm_counter = 0
        for i in range(self.n_jobs):
            self.permutation[:, perm_counter] = self.convert_bin_to_decimal(self.x[ : , i*self.n_bits: ((i+1)*self.n_bits)]).T
            perm_counter += 1

        # Normalize the job number
        self.permutation = self.permutation % self.n_jobs
        
        # The next section resolves repetitions to create a valid job sequence for the current machine
        for row_ind, job_seq in enumerate(self.permutation):
            # job_seq is the array of job numbers for the current machine (row_ind)
            indexes = []
            # This array will contain the missing values that should be replaced with the duplicates
            unique_vals = np.arange(0, self.n_jobs, dtype=int)
            for elem in range(len(job_seq)):
                # iterate over job numbers
                cur_val = job_seq[elem]
                if unique_vals[cur_val] != -1:
                    unique_vals[cur_val] = -1
                else:
                    # If the cur_val already was identified as redundant, add the reference to the value for the job_seq array
                    indexes.append(elem)

            index_counter = 0
            for missing_value in unique_vals:
                if missing_value != -1:
                    # If the element is not regarded as unique
                    self.permutation[row_ind][indexes[index_counter]] = missing_value
                    index_counter += 1
        
        # Merge the permutations
        self.permutation = self.permutation.ravel()
        return self.permutation
    

class QChromosomePositionEncoding(QChromosomeRepairPermutationEncoding):
    """This class consistutes the Quantum representation where the bit values are not thought about as job numbers but as position weights.
    This approach is very similar to the random key approach. The main difference is that for random key, the j indistinguishable job numbers 
    are placed in the position of the lowest j random keys. In this approach the indistinguishable numbers are placed periodically appart from 
    each other with the modulus operator. 

    Argsort retuns the indices that would sort the array.
    """
    
    def __init__(self, n_jobs: int, n_machines: int, individual_cfg: DictConfig, objectives: list, time_log: bool=False) -> None:
        self.individual_cfg = individual_cfg
        self.restrict_permutation = self.individual_cfg.restrict_permutation
        super().__init__(n_jobs, n_machines, individual_cfg, objectives, time_log)
        

    def convert_permutation(self) -> None:
        if self.restrict_permutation:
            self.convert_permutation_restricted()
        else:
            self.convert_permutation_full()

    @measure_runtime("Decode bit string")
    def convert_permutation_restricted(self):
        # This method produces n-repetition permutations by creating permutation for each machine and then merging them.

        # The binary values are not job numbers but the position weight of the corresponding jobnumber in the job sequence
        self.permutation = np.zeros((self.n_machines, self.n_jobs), dtype=int)

        # Convert from binary to decimal by working on each machine in parallel
        perm_counter = 0
        for i in range(self.n_jobs):
            self.permutation[:, perm_counter] = self.convert_bin_to_decimal(self.x[ : , i*self.n_bits: ((i+1)*self.n_bits)]).T
            perm_counter += 1

        # Normalize the job number
        # No need cut the job number range because 
        # self.permutation = self.permutation % self.n
        
        # The next section resolves repetitions to create a valid job sequence for the current machine
        for row_ind, position_weights in enumerate(self.permutation):
            self.permutation[row_ind] = np.argsort(position_weights)

        # Merge the permutations to create the n-repetition permutation
        self.permutation = self.permutation.ravel()
        return self.permutation
    
    @measure_runtime("Decode bit string")
    def convert_permutation_full(self):
        # This method produces n-repetition permutations directly.
        
        # The binary values are not job numbers but the position weight of the corresponding jobnumber in the job sequence
        self.permutation = np.zeros(self.n_machines*self.n_jobs, dtype=int)

        # Convert from binary to decimal
        x_flat = self.x.ravel()
        for i in range(self.n_jobs*self.n_machines):
            self.permutation[i] = self.convert_bin_to_decimal(x_flat[i*self.n_bits: ((i+1)*self.n_bits)].reshape(1,-1))

        # Normalize the job number
        # No need cut the job number range because 
        # self.permutation = self.permutation % self.n
        
        # The next section resolves repetitions to create a valid job sequence for the current machine
        self.permutation = np.argsort(self.permutation) % self.n_jobs

        return self.permutation
    

class QChromosomeHashPermutationEncoding(QChromosomeRepairPermutationEncoding):
    """This class consistutes the base for the hash method as decoding from binary to permutations. 
    The base class uses the method of splitting the problem into finding the job sequence for each 
    machine. Number of bits are m*j*(log_2(j-1)+1), which leads to redundency.

    The permutation number is found by using a periodic function. The resulting permutation number 
    is used to create the unique permutation amon the j! permutaitons. 

    Finaly, the machine sequences are merged to create the operation based representation.
    """
    
    def __init__(self, n_jobs: int, n_machines: int, individual_cfg: DictConfig, objectives: list, time_log: bool=False):
        super().__init__(n_jobs, n_machines, individual_cfg, objectives, time_log)

    @measure_runtime("Periodic mapping")
    def periodic_mapping_1(self, x, j):
        """This method is used to create a mapping between the binary strings and integer values representing the permutation number.
        This specific method constitutes a periodic function. The bitstrings that are considered equal are equally spaced apart in the numerical order.

        Parameters
        ----------
        x : np.array
            Array of integers constituting the binary values converted to decimal values. Run convert_bin_to_decimal() to convert bool array to int array
        j : Number of jobs 
            The specified number of jobs for the current problem

        Returns
        -------
        np.ndarray
            Array of permutation numbers
        """
        # Determine sign according to where the x is in the period. If x is in the last half, then sign := -1
        sign = (-1)**(np.floor(x / math.factorial(j)))
        down_slope = math.factorial(j)-1 - (x % math.factorial(j))
        up_slope = x % math.factorial(j)
        up_slope_contribution = (up_slope + sign*up_slope)/2
        down_slope_contribution = (down_slope - sign*down_slope)/2
        y = up_slope_contribution + down_slope_contribution
        return y

    @measure_runtime("Decode bit string")
    def convert_permutation(self):
        self.permutation = np.zeros((self.n_machines, self.n_jobs), dtype=int)
        # Convert all bit strings to decimal
        dec_values = self.convert_bin_to_decimal(self.x)
        # Obtain the mapping to permutation
        mapping_values = self.periodic_mapping_1(dec_values, self.n_jobs)
        for i, mapping in enumerate(mapping_values):
            self.permutation[i] = self._get_permutation(mapping, self.n_jobs)

        self.permutation = self.permutation.ravel()
        return self.permutation

    @measure_runtime("Get permutation")
    def _get_permutation(self, n: int, j: int):
        """This method is used to map an integer n to a permutation of j job numbers (j! possible permutations).
        Thus n must be between 0 and j!-1

        Parameters
        ----------
        n : int
            Permuation number
        j : int
            number of jobs (permutation size)
        """
        i = j
        p_next = n
        result = np.empty(i, dtype=int)
        selectables = np.arange(i)
        for k in range(len(result)):
            # Iterating over all elements that should be generated for the solution 
            j = i - k - 1
            cur_select_index = int(p_next / math.factorial(j))
            cur_job_number = selectables[selectables != -1][cur_select_index]
            result[k] = cur_job_number
            # Remove cur_job number form available jobs for next iteration
            selectables[cur_job_number] = -1
            p_next = p_next % math.factorial(j)

        return result

    
class QChromosomeHashMultisetEncoding(QChromosome):
    """This class reduces the search space for the bit string by estimating the number of bits more accurately.
    Each bit string maps to a permutation of 

    The permutation number is found by using a periodic function. The resulting permutation number 
    is used to create the unique permutation amon the j! permutaitons. 

    Finaly, the machine sequences are merged to create the operation based representation.
    """
    
    def __init__(self, n_jobs: int, n_machines: int, individual_cfg: DictConfig, objectives: list, time_log: bool=False):
        super().__init__(n_jobs, n_machines, objectives, time_log)

        # Determine how many bits are needed to represent the job number
        self.n_bits = self.calc_n_bits(n_jobs, n_machines) #int((self.n_jobs*self.n_machines*(np.log(self.n_machines*self.n_jobs) - np.log(self.n_jobs))/np.log(2))) #int(np.log2(self.n_jobs-1) + 1)
        # Create the amplitudes for the chromosome
        self.chromo_shape = (2, self.n_bits)
        self.binary_chromosome = np.ones(self.chromo_shape) # Dimensions: the number of amplitudes, machine number, number of bits for one job sequence
        # Set amplitudes to super position
        self.binary_chromosome = self.binary_chromosome * np.sqrt(2)**(-1)

        # Initialize mutation array
        self.mutation_flags = (np.random.uniform(0, 1, size=(self.n_bits)) < individual_cfg.mutation_rate).astype(int)

        # Create P(t) by first performing measurment and then convert permutations
        self.measure()
        self.convert_permutation()

    def calc_n_bits(self, j, m):
        """This function calculates the number of bits required for representation of the individual

        Parameters
        ----------
        j : int
            Number of jobs
        m : int
            Number of machines

        Returns
        -------
        int
            Number of bits in the bit string.
        """
        numerator = math.factorial(m * j)
        denominator = math.factorial(m) ** j
        value = (numerator // denominator) - 1  

        return math.floor(math.log2(value)) + 1

    @measure_runtime("Periodic mapping")
    def periodic_mapping_1(self, x, j, m):
        """This method is used to create a mapping between the binary strings and integer values representing the permutation number.
        This specific method constitutes a periodic function. The bitstrings that are considered equal are equally spaced apart in the numerical order.

        Parameters
        ----------
        x : np.array
            The decimal value corresponding to the bit string. Run convert_bin_to_decimal() to convert bool array to int array
        j : Number of jobs 
            The specified number of jobs for the current problem

        Returns
        -------
        np.ndarray
            Array of permutation numbers
        """
        # Determine sign according to where the x is in the period. If x is in the last half, then sign := -1
        max_permutation_number = math.factorial(m*j)//math.factorial(m)**j # ! This number grows very fast
        sign = (-1)**(np.floor(x / max_permutation_number))
        down_slope = max_permutation_number-1 - (x % int(max_permutation_number))
        up_slope = x % int(max_permutation_number)
        up_slope_contribution = decimal.Decimal(up_slope + sign*up_slope)/2
        down_slope_contribution = decimal.Decimal(down_slope - sign*down_slope)/2
        y = int(up_slope_contribution + down_slope_contribution)
        return y
    
    @measure_runtime("Find position")
    def find_position(self, i, n, r):
        # This function is called with the position number, the number of objects to fraw from and the number of objects to draw
        # The function should, based on position number, find the first value for r and return it
        # Starting with the first position and adding all possible positions in a loop
        # Base case:
        if r == 1:
            return (i, 0)
        
        r = r - 1
        cur_sum = 0
        # Start by placing the first r in position 0 thus n is reduced by a for each iteration
        for a in range(1, n):
            previous_sum = cur_sum
            cur_sum += math.factorial(n-a)/(math.factorial(r)*math.factorial(n-a-r))
            #print("Cur sum: " + str(cur_sum))
            if cur_sum > i:
                # If the sum is greater, we can asume that a-1 is the value of the first r
                # What happens if cur_sum == i: This would mean that the 
                next_number = i-previous_sum # int(i / previous_sum) if previous_sum > 0 else i
                return (a-1, int(next_number))
    
    @measure_runtime("get combination")
    def get_combination(self, n, r, combination_number):
        """Function that maps an integer (combination_number) to a combination of unique objects in a set of n objects

        Parameters
        ----------
        n : int
            set size
        r : int
            Combination size
        combination_number : int
        """
        result = np.empty(r, dtype=int) 
        previous_position = 0
        for i in range(r):
            next_r, combination_number = self.find_position(combination_number, n-previous_position, r-i)
            next_r += previous_position
            result[i] = next_r
            previous_position = next_r + 1

        return result

    @measure_runtime("Decode bit string")
    def convert_permutation(self):
        self.permutation = np.zeros((self.n_machines, self.n_jobs), dtype=int)
        # Convert all bit strings to decimal
        dec_values = self.convert_bin_to_decimal(self.x)
        # Obtain the mapping to permutation
        mapping_value = self.periodic_mapping_1(dec_values, self.n_jobs, self.n_machines)
        # Create the permutation
        self.permutation = self._get_permutation(mapping_value, self.n_jobs, self.n_machines)
        return self.permutation

    @measure_runtime("Get permutation")
    def _get_permutation(self, perm_number: int, j: int, m: int):
        """This method takes a permutation number as input and outputs a corresponding permutation with j*m objects where 
        each unique object j is repeated m times.

        Parameters
        ----------
        perm_number : int
            Permuation number
        j : int
            number of jobs (permutation size)
        m : int
            number of machines
        """
        # Initialize the array where the permutation is stored.
        # Note that the array elements are initialized with -1 to indicate that it contains no valid machine number.
        final_result = np.ones(m*j, dtype=int)*-1
        for k in range(j):
            # For each job
            cur_max = m*(j-k-1)
            if cur_max == 0:
                # Add the last job number in the remaining spaces
                final_result[final_result == -1] = k
            else:
                # Number of possible permutations is calculated by
                n_permutations = math.factorial(cur_max)//((math.factorial(m))**(j-k-1))
                # Find the number of times the number of permutations with j-1 goes in the permutations number
                cur_period_number = int(perm_number / n_permutations)
                # Find the new permutation number by finding the rest
                perm_number = int(perm_number) % n_permutations

                # Place the m elements of value j in the array.
                # This is done by finding the k-combination of an n-size integer set.
                temp_comb = self.get_combination(m*(j-k), m, cur_period_number)
                # Add the job numbers into the result array according to the combination created of index values
                temp_final_result = final_result[final_result == -1]
                temp_final_result[temp_comb] = k
                #for i in temp_comb:
                #    temp_final_result[i] = k

                final_result[final_result == -1] = temp_final_result
        
        return final_result

    def combinations(self, iterable, r, comb_number):
        # Function taken from https://docs.python.org/3/library/itertools.html#itertools.combinations
        # combinations('ABCD', 2) → AB AC AD BC BD CD
        # combinations(range(4), 3) → 012 013 023 123

        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        counter = 0
        cur_combination = tuple(pool[i] for i in indices)
        if counter == comb_number:
            return cur_combination
        
        counter += 1
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return cur_combination
            indices[i] += 1
            for j in range(i+1, r):
                # If i is not the last in the selection r, make sure to reset the indice to the right.
                indices[j] = indices[j-1] + 1
            cur_combination = tuple(pool[i] for i in indices)
            if counter == comb_number:
                return cur_combination
            
            counter += 1

    @measure_runtime("Measure qubits")
    def measure(self) -> np.ndarray:
        """This method executes the quantum measurement resulting in a bit string. 
           Overwrites the measure method in parent object QChromosome because the 
           dimensions of the qubit array is different.

        Returns
        -------
        pd.ndarray
            The measured bit string. Shape: n_machines, n_bits
        """
        cur_needles = np.random.uniform(0, 1, self.binary_chromosome.shape[1:])
        self.x = (self.binary_chromosome[0, :]**2 < cur_needles).astype(bool) # Note that x adopts the shape of the binary_chromosome except the amplitudes: machine numbers, n_bits
        # Add the static mutations
        self.x = np.abs((self.x - self.mutation_flags))
        return self.x
    
    def convert_bin_to_decimal(self, bin_array: np.ndarray) -> np.ndarray:
        """This method is used to convert the bit string to an integer

        Parameters
        ----------
        bin_array : np.array
            Array of boolean values representing the bit string

        Returns
        -------
        int
            The resulting integer
        """
        exponents = np.arange(-bin_array.shape[0] + 1, 1, dtype=object)
        bin_expo = exponents*-1 #(np.arange(-len(bin_array)+1, 1)*-1)
        result = np.sum(bin_array * (2**bin_expo))
        return result

class QChromosomeHashMultisetImprovedEncoding(QChromosome):
    """This class reduces the search space for the bit string by estimating the number of bits more accurately.
    Each bit string maps to a permutation of 

    The permutation number is found by using a periodic function. The resulting permutation number 
    is used to create the unique permutation amon the j! permutaitons. 

    Finaly, the machine sequences are merged to create the operation based representation.
    """
    
    def __init__(self, n_jobs: int, n_machines: int, individual_cfg: DictConfig, objectives: list, time_log: bool=False):
        super().__init__(n_jobs, n_machines, objectives, time_log)
        # Saving the schedule bounds from config as attributes
        #self.schedule_lb = individual_cfg.schedule_lb
        #self.schedule_ub = individual_cfg.schedule_ub
        self.reset_frequency = individual_cfg.reset_frequency
        # Determine how many bits are needed to represent the job number
        self.n_bits = self.calc_n_bits(n_jobs, n_machines) #int((self.n_jobs*self.n_machines*(np.log(self.n_machines*self.n_jobs) - np.log(self.n_jobs))/np.log(2))) #int(np.log2(self.n_jobs-1) + 1)
        # Create the amplitudes for the chromosome
        self.chromo_shape = (2, self.n_bits)
        self.binary_chromosome = np.ones(self.chromo_shape) # Dimensions: the number of amplitudes, machine number, number of bits for one job sequence
        # Set amplitudes to super position
        self.binary_chromosome = self.binary_chromosome * np.sqrt(2)**(-1)
        # Initialize mutation array
        self.mutation_flags = (np.random.uniform(0, 1, size=(self.n_bits)) < individual_cfg.mutation_rate).astype(int)
        # Create P(t) by first performing measurment and then convert permutations
        self.measure()
        self.convert_permutation()
        
    def calc_n_bits(self, j, m):
        """This function calculates the number of bits required for representation of the individual

        Parameters
        ----------
        j : int
            Number of jobs
        m : int
            Number of machines

        Returns
        -------
        int
            Number of bits in the bit string.
        """
        numerator = math.factorial(m * j)
        denominator = math.factorial(m) ** j
        self.n_permutations = numerator // denominator
        value = (self.n_permutations) - 1  

        return math.floor(math.log2(value)) + 1

    @measure_runtime("Periodic mapping")
    def periodic_mapping_1(self, x, j, m):
        """This method is used to create a mapping between the binary strings and integer values representing the permutation number.
        This specific method constitutes a periodic function. The bitstrings that are considered equal are equally spaced apart in the numerical order.

        Parameters
        ----------
        x : np.array
            The decimal value corresponding to the bit string. Run convert_bin_to_decimal() to convert bool array to int array
        j : Number of jobs 
            The specified number of jobs for the current problem

        Returns
        -------
        np.ndarray
            Array of permutation numbers
        """
        # Determine sign according to where the x is in the period. If x is in the last half, then sign := -1
        max_permutation_number = math.factorial(m*j)//math.factorial(m)**j # ! This number grows very fast
        sign = (-1)**(np.floor(x / max_permutation_number))
        down_slope = max_permutation_number-1 - (x % int(max_permutation_number))
        up_slope = x % int(max_permutation_number)
        up_slope_contribution = decimal.Decimal(up_slope + sign*up_slope)/2
        down_slope_contribution = decimal.Decimal(down_slope - sign*down_slope)/2
        y = int(up_slope_contribution + down_slope_contribution)
        return y
    
    @measure_runtime("Period mapping")
    def period_mapping(self, binary_value) -> int:
        return binary_value % self.n_permutations

    @measure_runtime("Uniform mapping")
    def uniform_mapping(self, binary_value) -> int:
        # Finding the number of redundant bit string for each permutation
        group_size = ((2**self.n_bits)-1) // self.n_permutations
        # Since the group size is rounded down, we need to find out how many bit strings are not covered with the current group size
        n_larger_groups = ((2**self.n_bits)-1) % self.n_permutations
        # The portion of the search space wich should have the larger groups (group_size+1)
        large_groups_partition = n_larger_groups*(group_size + 1)
        if binary_value <= large_groups_partition:
            # If the binary_value is in the part where the larger group sizes are defined, 
            # returning the number of large groups within the binary value gives the coresponding permutation rank
            return binary_value // (group_size + 1)
        else: 
            # If the binary_value is in the last part, offset the obtained rank with the number of binary values covered by the prio partition
            return n_larger_groups + ((binary_value - large_groups_partition) // group_size)

    @measure_runtime("Find position")
    def get_combination(self, n, k, rank):
        """This method uses an algorithm to unrank combinations in colexicographic order.
        

        Parameters
        ----------
        n : int
            Size of set to draw from
        k : int
            Number of elements to draw from set
        rank : int
            The rank of the combination to create

        Returns
        -------
        np.array
            The array containing the combination
        """
        res = np.empty(k, dtype=int)
        res_counter = 0
        i = k
        cur_rank = rank
        while i >= 1:
            p = self.get_p(n, i, cur_rank)
            cur_rank = cur_rank - scipy.special.binom(p-1, i)
            res[res_counter] = p
            i -= 1
            res_counter += 1

        return res
    
    def get_p(self, n, i, r):
        """This method is used to optimize the binomial coefficient with regards to the rank r. 
        For C(n, i), n is minimized with the constraint C(n, r) > r.
        The optimization is performed with binary search.

        Parameters
        ----------
        n : int
            Size of set to draw i elements from
        i : int
            Number of unique elements to draw from the set of size n
        r : int
            The rank of the combination

        Returns
        -------
        int
            The optimized n value (cur_p identifier is used in the algorithm)
        """
        if r <= 0:
            # If the rank is 0 
            return i
        # The upper bound (UB) for r = C(p, i)
        upper_n = np.floor(i*(r**(1/i)))
        
        # If the upper bound is less than i, then p must be i+1 because C(i-1, i) = 1/i(-1)! -> not defined
        if upper_n < i:
            return i+1
        
        # Check if the upper bound is too low
        if scipy.special.binom(upper_n, i) <= r:
            return upper_n + 1
        
        # The lower bound (LB)
        lower_n = np.ceil((r*(math.factorial(i)))**(1/i))

        #if i > lower_n:
        #    lower_n = i

        if upper_n == lower_n:
            # If there is no difference between bounds, check that the bound satisfies the constraint
            if scipy.special.binom(upper_n, i) > r: # Cannot happen, because there 
                return upper_n
            else:
                # The soght after p value must be the next integer
                return upper_n + 1
            
        # Binary search for the correct p value
        counter = 0
        while True:
            if counter > 10000:
                print("Debug case")

            # Select a p value in the middle of the current range  
            cur_p: int = lower_n + int((upper_n - lower_n)/2)
            # Calculate the binomial coefficient with the selected p value
            cur_binom: int = scipy.special.binom(cur_p, i)
            if cur_binom > r:
                # The binomial coefficient is greater than the rank
                # find binom(p-1, i)
                lower_p_binom = cur_binom*(cur_p-i)/cur_p
                if lower_p_binom <= r:
                    # If the lower binomial coefficient is lower, then the optimal was found and is returned
                    return cur_p
                else:
                    # The upper bound is updated since there is at least one coefficient that is smaller than r
                    upper_n = cur_p
            elif cur_binom <= r:
                # The p value is too low, updating lower bound
                lower_n = cur_p + 1
            
            counter += 1


    @measure_runtime("Decode bit string")
    def convert_permutation(self):
        self.permutation = np.zeros((self.n_machines, self.n_jobs), dtype=int)
        # Convert all bit strings to decimal
        dec_value = int(gray_to_bin("".join(self.x.astype(str))), 2) #self.convert_bin_to_decimal(self.x)
        # Obtain the mapping to permutation
        self.mapping_value = self.uniform_mapping(dec_value) #self.periodic_mapping_1(dec_value, self.n_jobs, self.n_machines)
        # Create the permutation
        self.permutation = self.get_permutation(self.mapping_value, self.n_jobs, self.n_machines)
        return self.permutation

    @measure_runtime("Get permutation")
    def get_permutation(self, perm_number: int, j: int, m: int):
        """This method takes a permutation number as input and outputs a corresponding permutation with j*m objects where 
        each unique object j is repeated m times.

        Parameters
        ----------
        perm_number : int
            Permuation number
        j : int
            number of jobs (permutation size)
        m : int
            number of machines
        """
        # Initialize the array where the permutation is stored.
        # Note that the array elements are initialized with -1 to indicate that it contains no valid machine number.
        final_result = np.ones(m*j, dtype=int)*-1
        original_perm_number = perm_number
        for k in range(j):
            #if k == 17 and debug:
            #    print("breakpoint")
            # For each job
            cur_max = int(m*(j-k-1))
            if cur_max == 0:
                # Add the last job number in the remaining spaces
                final_result[final_result == -1] = k
            else:
                # Number of possible permutations is calculated by
                n_permutations = math.factorial(cur_max)//((math.factorial(m))**(j-k-1))
                # Find the number of times the number of permutations with j-1 goes in the permutations number
                cur_period_number = perm_number // n_permutations
                # Find the next permutation number by finding the rest
                perm_number = perm_number % n_permutations

                # Place the m elements of value j in the array.
                # This is done by finding the k-combination of an n-size integer set.
                temp_comb = self.get_combination(m*(j-k), m, cur_period_number)-1
                # Add the job numbers into the result array according to the combination created of index values
                original_final_result = copy.deepcopy(final_result)
                temp_final_result = final_result[final_result == -1]
                temp_final_result[temp_comb] = k
                final_result[final_result == -1] = temp_final_result
        
        # test the validity of the permutation
        #if len(np.unique(np.bincount(final_result))) > 1:
        #    raise Exception(f"The created permutation is not valid: incorect number of job occurences. \n {np.bincount(final_result)}")
        return final_result


    @measure_runtime("Measurment")
    def measure(self) -> np.ndarray:
        """This method executes the quantum measurement resulting in a bit string.

        Returns
        -------
        pd.ndarray
            The measured bit string. Shape: n_machines, n_bits
        """
        cur_needles = np.random.uniform(0, 1, self.binary_chromosome.shape[1:])
        self.x = (self.binary_chromosome[0, :]**2 < cur_needles).astype(bool) # Note that x adopts the shape of the binary_chromosome except the amplitudes: machine numbers, n_bits
        # Add the static mutations
        self.x = np.abs((self.x - self.mutation_flags))
        return self.x
    
    def convert_bin_to_decimal(self, bin_array: np.ndarray) -> np.ndarray:
        """This method is used to convert the bit string to an integer

        Parameters
        ----------
        bin_array : np.array
            Array of boolean values representing the bit string

        Returns
        -------
        int
            The resulting integer
        """
        exponents = np.arange(-bin_array.shape[0] + 1, 1, dtype=object)
        bin_expo = exponents*-1 #(np.arange(-len(bin_array)+1, 1)*-1)
        result = np.sum(bin_array * (2**bin_expo))
        return result
    
    def rotate(self, 
               b: object, 
               c: int,
               c_tot: int,
               n_groups: int,
               cur_group : int,
               rotation_angles: str="[0.2*np.pi, 0, 0.5*np.pi, 0, 0.5*np.pi, 0, 0.2*np.pi, 0]"
               ):
        """This method executes the rotation operation to perturb the amplitudes of the qubits constituting the quantum chromosome.
        It is similar to the funciton in QChromosome, but handles the bits differently depending on their significants towards the binary value

        Parameters
        ----------
        b : object
            The solution to compare to
        c : int
            The current generation number
        c_tot : int
            The total generations to run 
        n_groups : int
            Number of groups that solutions are divided into
        cur_group : int
            The id of the group that the current idividual belongs to.
        restart_frequency : int
            The number of times the individual is reset and the cover schedule is restarted
        rotation_angles : str, optional
            The rotation angles for each combination of b and x bit, by default "[0.2*np.pi, 0, 0.5*np.pi, 0, 0.5*np.pi, 0.5*np.pi, 0, 0.2*np.pi]"

        Raises
        ------
        ValueError
            If the rotation angle does not contain 8 values.
        """
        # Find the static portion for this individual
        #lb = lambda c: (c)/(c_tot)
        #ub = lambda c: np.log(c)/np.log(c_tot)
        c = np.array([c])
        reset_iterations = np.floor(c_tot/self.reset_frequency)
        cur_periodic_c = c % reset_iterations
        if cur_periodic_c <= 0:
            cur_periodic_c = np.array([1])
        schedule_development = lambda c: np.log(cur_periodic_c)/np.log((reset_iterations)*0.8)

        #if c <= 0:
        #    cur_ub = self.schedule_ub
        #else:    
        #    cur_ub = ub(c)

        #cur_lb = lb(c)
        #cur_ub[np.where(cur_ub < self.schedule_ub)[0]] = self.schedule_ub
        #cur_lb[np.where(cur_lb < self.schedule_lb)[0]] = self.schedule_lb
        #d = (cur_ub - cur_lb)/(n_groups-1)
        static_portion = schedule_development(c)
        static_portion[np.where(static_portion > 0.98)[0]] = 0.98

        #static_portion = cur_ub - (cur_group)*d

        cached_shape = b.x.shape
        cutoffpoint = int(np.floor(cached_shape[0]*static_portion))
        cur_b = b.x[cutoffpoint:]
        cur_x = self.x[cutoffpoint:]
        
        if c % reset_iterations == 0:
            # reset chromosome
            self.binary_chromosome[0, : ] = np.sqrt(2)**(-1)
            self.binary_chromosome[1, : ] = np.sqrt(2)**(-1)

        else:
            # Test that the rotation angles are valid
            raw_rotation_angles = eval(rotation_angles)
            if len(raw_rotation_angles) != 8:
                raise ValueError("Please specify rotation angle array of length 8.")

            rotation_angles = np.array(raw_rotation_angles) #np.repeat([raw_rotation_angles], self.n_bits*self.n_jobs*self.n_machines, axis=0)
            #rotation_angles = rotation_angles.reshape(self.n_machines, self.n_bits*self.n_jobs, -1)

            signs = np.array([-1, 0, 1, 0, -1, 0, 1, 0]) #np.repeat([np.array([-1, 0, 1, 0, -1, 0, 1, 0])], self.n_bits*self.n_jobs*self.n_machines, axis=0)
            #signs = signs.reshape(self.n_machines, self.n_jobs*self.n_bits, -1)
            #for i in range(len(cur_x)):
            pi = cur_x.astype(int) * (2**2)
            bi = cur_b.astype(int) * (2**1)
            # the best individual has always a better fintess in  this case
            better = int(False) 
            index = (pi + bi + better).ravel() #int(str(pi) + str(bi) + str(better), 2)
            cur_sign = (self.binary_chromosome[0, cutoffpoint:] * self.binary_chromosome[1, cutoffpoint:]) < 0
            cur_angle = rotation_angles[index] * signs[index] * (((-2)*cur_sign.astype(int).ravel())+1)
            cur_angle = cur_angle.reshape(cur_b.shape)
            # Apply the rotation
            new_a = self.binary_chromosome[0, cutoffpoint:]*np.cos(cur_angle) - self.binary_chromosome[1, cutoffpoint:]*np.sin(cur_angle)
            new_b = self.binary_chromosome[0, cutoffpoint:]*np.sin(cur_angle) + self.binary_chromosome[1, cutoffpoint:]*np.cos(cur_angle)

            self.binary_chromosome[0, cutoffpoint:] = new_a
            self.binary_chromosome[1, cutoffpoint:] = new_b
            # Converge more significant bits
            self.binary_chromosome[0, :cutoffpoint] = np.logical_not(b.x[:cutoffpoint])
            self.binary_chromosome[1, :cutoffpoint] = b.x[:cutoffpoint]


class EnhancedQuantumRandomKeyIndividual(Individual):
    
    def __init__(self, n_jobs: int, n_machines: int, individual_cfg: DictConfig, objectives: list, time_log: bool=False) -> None:
        super().__init__(n_jobs, n_machines, objectives, time_log)
        self.individual_cfg = individual_cfg
        self.start_std = self.individual_cfg.start_std
        self.base_permutation = np.repeat(np.arange(self.n_jobs), self.n_machines)
        self.initialize_individual()
        self.convert_permutation()

    def initialize_individual(self):
        self.positions = np.random.randint(0, self.n_jobs, size=self.n_jobs*self.n_machines)
        self.standard_deviations = np.ones(shape=self.n_jobs*self.n_machines) * self.start_std
        self.random_keys = np.zeros_like(self.positions)

    def periodic_triangular_function_vectorized(self, x: np.ndarray, j: int) -> np.ndarray:
        y = lambda x, offset, sign: sign*(x - offset)
        y_res = np.empty_like(x)
        
        x1 = x[(np.floor(x) // j) % 2 == 0]
        x2 = x[(np.floor(x) // j) % 2 > 0]

        y_res[(np.floor(x) // j) % 2 == 0] = y(x1, (np.floor(x1) // j)*j, 1)
        y_res[(np.floor(x) // j) % 2 > 0] = y(x2, ((np.floor(x2) // j)*j)+j, -1)

        return np.floor(y_res)
    
    def measure(self):
        self.random_keys = self.periodic_triangular_function_vectorized(np.round(np.random.normal(0, self.standard_deviations) + self.positions).astype(int), self.n_jobs-1)

    def convert_permutation(self): 
        self.permutation = self.base_permutation[np.argsort(self.random_keys)]

    def rotate(self, 
               b: object, 
               c: int,
               c_tot: int,
               n_groups: int,
               cur_group : int,
               rotation_angles: str="[0, 0, 1, 0, 0, 0, 1, 0]",
               std_deltas: str="[-0.3, -0.1, -0.3, -0.1]"):
        
        # transfer all angles into interval [0, j]
        raw_rotation_angles = eval(rotation_angles)
        if len(raw_rotation_angles) != 8:
            raise ValueError("Please specify rotation angle array of length 8.")
        
        raw_std_deltas = eval(std_deltas)
        if len(raw_std_deltas) != 4:
            raise ValueError("Please specify std delta array of length 4.")

        rotation_angles = np.array(raw_rotation_angles)
        std_deltas = np.array(raw_std_deltas)

        b_keys = b.random_keys
        x_keys = self.random_keys
        x_filtered = self.positions
        # Check if the two angles are within the variance
        equal_cases_overshoot = np.logical_and(b_keys == x_keys, (b_keys - x_keys) <= 0)
        equal_cases_undershoot = np.logical_and(b_keys == x_keys, (b_keys - x_keys) >= 0)
        unequal_cases_overshoot = np.logical_and(b_keys != x_keys, (b_keys - x_keys) < 0)
        unequal_cases_undershoot = np.logical_and(b_keys != x_keys, (b_keys - x_keys) > 0)

        x_filtered[equal_cases_overshoot] += rotation_angles[0] #np.random.uniform(0.01, 0.05)
        x_filtered[unequal_cases_overshoot] -= rotation_angles[2] #np.random.uniform(0.01, 0.05)
        x_filtered[equal_cases_undershoot] -= rotation_angles[4] #np.random.uniform(0.01, 0.05)
        x_filtered[unequal_cases_undershoot] += rotation_angles[6] #np.random.uniform(0.01, 0.05)
        # Contribute to std convergence
        self.standard_deviations[equal_cases_overshoot] = np.abs(self.standard_deviations[equal_cases_overshoot] + std_deltas[0])
        self.standard_deviations[unequal_cases_overshoot] = np.abs(self.standard_deviations[unequal_cases_overshoot] + std_deltas[1])
        self.standard_deviations[equal_cases_undershoot] = np.abs(self.standard_deviations[equal_cases_undershoot] + std_deltas[2])
        self.standard_deviations[unequal_cases_undershoot] = np.abs(self.standard_deviations[unequal_cases_undershoot] + std_deltas[3])
        # If the standard deviaiton has become negative, make sure it is 
        #self.standard_deviations[self.standard_deviations < 0] = 0
        # Handle values outside the supported range
        self.positions = np.abs(x_filtered) % self.n_jobs
        return self.permutation

       

if __name__=="__main__":
    m = j = 20
    start = time.time()
    test_chromo_full = QChromosomePositionEncoding(j, m, DictConfig(content={"restrict_permutation" : True}), time_log=True)
    print(f"Time chromo_full: {time.time()-start:.4f}")
    
    start = time.time()
    test_chromo_restricted = QChromosomePositionEncoding(j, m, DictConfig(content={"restrict_permutation" : False}), time_log=True)
    print(f"Time chromo_restricted: {time.time()-start:.4f}")
    
    start = time.time()
    test_chromo_hash = QChromosomeHashPermutationEncoding(j, m, {}, time_log=True)
    print(f"Time chromo_hash: {time.time()-start:.4f}")
    
    start = time.time()
    test_chromo_hash_II = QChromosomeHashMultisetImprovedEncoding(j, m, {}, time_log=True)
    print(f"Time chromo_hash_II: {time.time()-start:.4f}")


    #print(test_chromo_full.timer_result)
    #print(test_chromo_restricted.timer_result)
    #print(test_chromo_hash.timer_result)
    print(test_chromo_hash_II.timer_result)
    #print(test_chromo_restricted.permutation)
    #print(test_chromo_hash.permutation)
    print(test_chromo_hash_II.permutation)