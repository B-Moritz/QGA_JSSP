# This file contains the object representing an individual/chromosome
import numpy as np
import math

from operations import Operation
from schedules import Schedule
from direct_representations import *

from typing import List


class Individual:

    def __init__(self, n_jobs: int, n_machines: int):
        self.job_permutation = np.array([])
        self.n_machines: int = n_machines
        self.n_jobs: int = n_jobs
    
    def create_schedule(self, 
                        method: str, 
                        jssp_problem: np.ndarray,
                        job_permutation: np.ndarray,
                        #objectives: List[str],
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

        if len(self.job_permutation) == 0:
            # Make sure a job permutation is available
            raise Exception("Please create the n-repetition permutation before attempting to create the schedule.")
        
        # Create operation list
        operation_list = encoding_method(
            self.n_jobs, 
            self.n_machines,
            self.job_permutation,
            jssp_problem
        )
        # Create schedule
        self.schedule = Schedule(operation_list, self.n_jobs, self.n_machines, jssp_problem)
        # Activate schedule
        if activate_schedule:
            self.schedule.activate_schedule()

        # Make fitness evaluations
        self.cur_fitness = np.array([self.schedule.get_makespan(), self.schedule.get_mean_flow_time()])


class Permutation_chromosome(Individual):

    def __init__(self, permutation: np.ndarray, n_jobs: int, n_machines: int):
        super().__init__(permutation, n_jobs, n_machines)
        
        if len(permutation) != self.n_jobs*self.n_machines:
            raise Exception("Bad arguments for number of jobs and machines")

        self.job_permutation: np.ndarray = permutation

    def Permutation_chromosome(n_jobs: int, n_machines: int):
        # Generator funciton for creating an individual object
        permutation = create_m_rep_permutation(n_jobs, n_machines)
        return Individual(permutation, n_jobs, n_machines)




class QChromosome(Individual):

    def __init__(self, n_jobs : int, n_machines : int, conversion_method="Old") -> None:
        super().__init__(n_jobs, n_machines)

        #conversion_dict = {
        #    "Old" : self.convert_permutation, 
        #    "New" : self.convert_permutation_new, 
        #    "Hashing" : self.convert_permutation_hashing}

    def measure(self) -> np.ndarray:
        """This method executes the quantum measurement resulting in a bit string.

        Returns
        -------
        pd.ndarray
            The measured bit string. Shape: n_machines, n_bits
        """
        cur_needles = np.random.uniform(0, 1, self.binary_chromosome.shape[1:])
        self.x = (self.binary_chromosome[0, :, :]**2 < cur_needles).astype(bool) # Note that x adopts the shape of the binary_chromosome except the amplitudes: machine numbers, n_bits
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
    

    def rotate(self, b : object):
        cur_b = b.x
        cur_x = self.x
        rotation_angles = np.array([0.2*np.pi, 0, 0.5*np.pi, 0, 0.5*np.pi, 0.5*np.pi, 0, 0.2*np.pi])
        signs = np.array([-1, 0, 1, 0, -1, 0, 1, 0])
        for i in range(len(cur_x)):
            pi = cur_x[:, i]
            bi = cur_b[:, i]
            better = int(True)
            index = int(str(pi) + str(bi) + str(better), 2)
            cur_sign = self.binary_chromosome[0, :, i] * self.beta_amplitudes[1, :, i]
            if cur_sign > 0:
                cur_angle = rotation_angles[index] * signs[index]
            else:
                cur_angle = rotation_angles[index] * signs[index] * -1

            # Apply the rotation
            new_a = np.cos(cur_angle) - np.sin(cur_angle)
            new_b = np.sin(cur_angle) + np.cos(cur_angle)

            self.binary_chromosome[0, :, i] = new_a
            self.binary_chromosome[0, :, i] = new_b

    
class QChromosomeBaseEncoding(QChromosome):
    
    def __init__(self, n_jobs : int, n_machines : int):
        super().__init__(n_jobs, n_machines)

        # Determine how many bits are needed to represent the job number
        self.n_bits = int(np.log2(self.n_jobs-1) + 1)
        # Create the aplitudes for the chromosome
        self.length = self.n_bits*self.n_jobs
        self.binary_chromosome = np.ones((2, self.n_machines, self.n_bits*self.n_jobs)) # Dimensions: the number of amplitudes, machine number, number of bits for one job sequence
        # Set amplitudes to super position
        self.binary_chromosome = self.binary_chromosome * np.sqrt(2)**(-1)

        # Create P(t) by first performing measurment and then convert permutations
        self.measure()
        self.convert_permutation()

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
    

class QChromosomePositionEncoding(QChromosomeBaseEncoding):
    """This class consistutes the Quantum representation where the bit values are not thought about as job numbers but as position weights.
    This approach is very similar to the random key approach. The main difference is that for random key, the j indistinguishable job numbers 
    are placed in the position of the lowest j random keys. In this approach the indistinguishable numbers are placed periodically appart from 
    each other with the modulus operator. 

    Argsort retuns the indices that would sort the array.
    """
    
    def __init__(self, n_jobs : int, n_machines : int, restrict_permutation: bool=False) -> None:
        self.restrict_permutation = restrict_permutation
        super().__init__(n_jobs, n_machines)
        

    def convert_permutation(self) -> None:
        if self.restrict_permutation:
            self.convert_permutation_restricted()
        else:
            self.convert_permutation_full()

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
    

class QChromosomeHashBaseEncoding(QChromosome):
    """This class consistutes the base for the hash method as decoding from binary to permutations. 
    The base class uses the method of splitting the problem into finding the job sequence for each 
    machine. Number of bits are m*j*(log_2(j-1)+1), which leads to redundency.

    The permutation number is found by using a periodic function. The resulting permutation number 
    is used to create the unique permutation amon the j! permutaitons. 

    Finaly, the machine sequences are merged to create the operation based representation.
    """
    
    def __init__(self, n_jobs : int, n_machines : int):
        super().__init__(n_jobs, n_machines)

        # Determine how many bits are needed to represent the job number
        self.n_bits = int(np.log2(self.n_jobs-1) + 1)
        # Create the aplitudes for the chromosome
        self.length = self.n_bits*self.n_jobs
        self.binary_chromosome = np.ones((2, self.n_machines, self.n_bits*self.n_jobs)) # Dimensions: the number of amplitudes, machine number, number of bits for one job sequence
        # Set amplitudes to super position
        self.binary_chromosome = self.binary_chromosome * np.sqrt(2)**(-1)

        # Create P(t) by first performing measurment and then convert permutations
        self.measure()
        self.convert_permutation()

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
            j = i - k - 1
            cur_select_index = int(p_next / math.factorial(j))
            cur_job_number = selectables[selectables != -1][cur_select_index]
            result[k] = cur_job_number
            # Remove cur_job number form available jobs for next iteration
            selectables[cur_job_number] = -1
            p_next = p_next % math.factorial(j)

        return result
    

    
class QChromosomeHashReducedEncoding(QChromosome):
    """This class reduces the search space for the bit string by estimating the number of bits more accurately.
    Each bit string maps to a permutation of 

    The permutation number is found by using a periodic function. The resulting permutation number 
    is used to create the unique permutation amon the j! permutaitons. 

    Finaly, the machine sequences are merged to create the operation based representation.
    """
    
    def __init__(self, n_jobs : int, n_machines : int):
        super().__init__(n_jobs, n_machines)

        # Determine how many bits are needed to represent the job number
        self.n_bits = int((self.n_jobs*self.n_machines*(np.log(self.n_machines*self.n_jobs) - np.log(self.n_jobs))/np.log(2))) #int(np.log2(self.n_jobs-1) + 1)
        # Create the amplitudes for the chromosome
        self.binary_chromosome = np.ones((2, self.n_bits)) # Dimensions: the number of amplitudes, machine number, number of bits for one job sequence
        # Set amplitudes to super position
        self.binary_chromosome = self.binary_chromosome * np.sqrt(2)**(-1)

        # Create P(t) by first performing measurment and then convert permutations
        self.measure()
        self.convert_permutation()

    def periodic_mapping_1(self, x, j, m):
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
        max_permutation_number = math.factorial(m*j)/math.factorial(j)**m # ! This number grows very fast
        sign = (-1)**(np.floor(x / max_permutation_number))
        down_slope = max_permutation_number-1 - (x % max_permutation_number)
        up_slope = x % max_permutation_number
        up_slope_contribution = (up_slope + sign*up_slope)/2
        down_slope_contribution = (down_slope - sign*down_slope)/2
        y = up_slope_contribution + down_slope_contribution
        return y

    def convert_permutation(self):
        self.permutation = np.zeros((self.n_machines, self.n_jobs), dtype=int)
        # Convert all bit strings to decimal
        dec_values = self.convert_bin_to_decimal(self.x)
        # Obtain the mapping to permutation
        mapping_value = self.periodic_mapping_1(dec_values, self.n_jobs, self.n_machines)
        # Create the permutation
        self.permutation = self._get_permutation(mapping_value, self.n_jobs, self.n_machines)
        return self.permutation

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
        
        final_result = np.ones(m*j, dtype=int)*-1
        for k in range(j):
            # For each job
            cur_max = m*(j-k-1)
            if cur_max == 0:
                # Add the last job number in the remaining spaces
                final_result[final_result == -1] = k
            else:
                cur_count = math.factorial(cur_max)/(math.factorial(m))**(j-k-1)
                
                cur_period_number = int(perm_number / cur_count)
                perm_number = perm_number % cur_count

                temp_comb = self.combinations(range(m*(j-k)), m, cur_period_number)
                # Add the job numbers into the permutation array
                temp_final_result = final_result[final_result == -1]
                for i in temp_comb:
                    temp_final_result[i] = k

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

    def measure(self) -> np.ndarray:
        """This method executes the quantum measurement resulting in a bit string.

        Returns
        -------
        pd.ndarray
            The measured bit string. Shape: n_machines, n_bits
        """
        cur_needles = np.random.uniform(0, 1, self.binary_chromosome.shape[1:])
        self.x = (self.binary_chromosome[0, :]**2 < cur_needles).astype(bool) # Note that x adopts the shape of the binary_chromosome except the amplitudes: machine numbers, n_bits
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
        exponents = np.arange(-bin_array.shape[0]+1, 1)
        bin_expo = exponents*-1 #(np.arange(-len(bin_array)+1, 1)*-1)
        result: np.ndarray = np.sum(bin_array * 2**bin_expo)
        return result
    
    

if __name__=="__main__":
    #test_chromo_full = QChromosomePositionEncoding(3, 3)
    #test_chromo_restricted = QChromosomePositionEncoding(3, 3, restrict_permutation=True)
    #test_chromo_hash = QChromosomeHashBaseEncoding(3, 3)
    test_chromo_hash_II = QChromosomeHashReducedEncoding(3, 3)


    #print(test_chromo_full.permutation)
    #print(test_chromo_restricted.permutation)
    #print(test_chromo_hash.permutation)
    print(test_chromo_hash_II.permutation)