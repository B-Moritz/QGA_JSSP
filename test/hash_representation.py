
import numpy as np
import decimal
import math
import pdb
import scipy

def periodic_mapping_1(x, j, m):
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
    max_permutation_number = decimal.Decimal(math.factorial(m*j))/decimal.Decimal(math.factorial(m)**j) # ! This number grows very fast
    sign = (-1)**(np.floor(x / max_permutation_number))
    down_slope = max_permutation_number-1 - (x % int(max_permutation_number))
    up_slope = x % int(max_permutation_number)
    up_slope_contribution = decimal.Decimal(up_slope + sign*up_slope)/2
    down_slope_contribution = decimal.Decimal(down_slope - sign*down_slope)/2
    y = int(up_slope_contribution + down_slope_contribution)
    return y

def get_p(n, k, r):
    """This method is used to optimize the binomial coefficient with regards to the rank r. 
    For C(n, k), n is minimized with the constraint C(n, r) > r.
    The optimization is performed with binary search.

    Parameters
    ----------
    n : int
        Size of set to draw k elements from
    k : int
        Number of unique elements to draw from the set of size n
    r : int
        The rank of the combination

    Returns
    -------
    int
        The optimized n value (cur_p identifier is used in the algorithm)
    """
    if r <= 0:
        return k
    # Calculate bounds
    upper_n = np.floor((r*(k**k))**(1/k))
    # If the upper bound is less than k, then p must be k+1
    if upper_n < k:
        return k+1
    
    #if n < upper_n:
    #    upper_n = n
    
    lower_n = np.ceil((r*(math.factorial(k)))**(1/k))
    #if k > lower_n:
    #    lower_n = k

    if upper_n == lower_n:
        if scipy.special.binom(upper_n, k) > r:
            return upper_n
        else:
            return upper_n + 1
    # Binary search for the correct p value
    counter = 0
    while True:
        if counter > 10000:
            print("Debug case")
            
        cur_p = lower_n + int((upper_n - lower_n)/2)
        cur_binom = scipy.special.binom(cur_p, k)
        if cur_binom > r:
            # The binomial coefficient is greater than the rank
            # find binom(p-1, k)
            lower_p_binom = cur_binom*(cur_p-k)/cur_p
            if lower_p_binom <= r:
                # If the lower binomial coefficient is lower, then the optimal was found and is returned
                return cur_p
            else:
                # The upper bound is updated since there is at least one coefficient that is smaller than r
                upper_n = cur_p
        if cur_binom <= r:
            # The p value is too low, updating lower bound
            lower_n = cur_p + 1
        
        counter += 1

def get_combination(n, k, rank):
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
        p = get_p(n, i, cur_rank)
        cur_rank = cur_rank - scipy.special.binom(p-1, i)
        res[res_counter] = p
        i -= 1
        res_counter += 1

    return res
       
def get_permutation(perm_number: int, j: int, m: int):
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
            n_permutations = int(decimal.Decimal(math.factorial(cur_max))/decimal.Decimal((math.factorial(m))**(j-k-1)))
            # Find the number of times the number of permutations with j-1 goes in the permutations number
            cur_period_number = int(perm_number / n_permutations)
            # Find the new permutation number by finding the rest
            perm_number = int(perm_number) % n_permutations

            # Place the m elements of value j in the array.
            # This is done by finding the k-combination of an n-size integer set.
            temp_comb = get_combination(m*(j-k), m, cur_period_number)-1
            # Add the job numbers into the result array according to the combination created of index values
            temp_final_result = final_result[final_result == -1]
            temp_final_result[temp_comb] = k
            #for i in temp_comb:
            #    temp_final_result[i] = k

            final_result[final_result == -1] = temp_final_result
    
    return final_result


if __name__=="__main__":
    j = 5
    m = 10
    n_bits = 2**(int((j*m*(np.log(m*j) - np.log(j))/np.log(2))))-1
    print(int((j*m*(np.log(m*j) - np.log(j))/np.log(2))))
    cur_perm_num = periodic_mapping_1(n_bits-10000, j, m)
    max_rank = int(math.factorial(m*j)/decimal.Decimal(math.factorial(j)**m))
    for i in range(max_rank):
        print(get_permutation(max_rank-10-i, j, m))
