# This file contains the implementation of the QEA presented by (Han & Kim, 2002):
# Han, K.-H., & Kim, J.-H. (2002). Quantum-inspired evolutionary algorithm for a class of combinatorial optimization. IEEE Transactions on Evolutionary Computation, 6(6), 580–593. https://doi.org/10.1109/TEVC.2002.804320


import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt
import copy
import time
import pdb
import os
import subprocess

from  matplotlib import cm
from matplotlib.colors import Normalize



class Gene:
    """This class represents the registry of qubits that represents one element of the genotype encoding in a chromosome.
    """
    #lookup_table = np.array([0, 0, 0.01*np.pi, 0, -0.01*np.pi, 0, 0, 0])
    lookup_table = np.array([0, 0, 0.01*np.pi, 0, -0.01*np.pi, 0, 0, 0])
    def __init__(self, register_size=1) -> None:
        self.register_size = register_size
        self.q_register = np.ones((2, self.register_size)) # first row represents |0> and second |1>
        self.q_register[:, :] = 1/np.sqrt(2)
        
    def decode_gene(self) -> int:
        # Pick a random number from the uniform distribution between 0 and 1 for all qubits in gene
        random_values = np.random.uniform(low=0, high=1, size=self.register_size)
        result_bit_string = random_values < abs(self.q_register[0,:])**2
        #convert to decimal
        self.decimal_result = int("".join(result_bit_string.astype("int").astype("str")), 2)
        return self.decimal_result

    def initialize_gene(register_size=1):
        """Generator function to generate Gene objects initialized with 1/sqrt(2) as apmplitudes

        Parameters
        ----------
        register_size : int, optional
            Size of quantumregister , by default 1

        Returns
        -------
        Gene
        """
        return Gene(register_size)
    
    def update_register(self, bi, x_profit, b_profit):
        """This method performs the update to the qubit amplitudes by applying a pauli y gate. 

        Parameters
        ----------
        bi : int
            bit value for the ith element in the best solution
        x_profit : int
            The measured profit of x
        b_profit : int
            THe measured profit of the best solution
        """

        cur_profit_comp = x_profit >= b_profit
        # Create index for lookup
        bit_string = self.decimal_result*2**2 + bi*2**1 + int(cur_profit_comp)*2**0
        cur_rotation_angle = self.lookup_table[int(bit_string)]
        for qubit_index in range(self.register_size):
            temp_alpha = self.q_register[0, qubit_index]
            temp_beta = self.q_register[1, qubit_index]
            if temp_alpha*temp_beta > 0:
                # If in second or fourth quadrant, then the rotation angle must be inverted
                cur_rotation_angle = cur_rotation_angle * -1

            self.q_register[0, qubit_index] = np.cos(cur_rotation_angle)*temp_alpha - np.sin(cur_rotation_angle)*temp_beta
            self.q_register[1, qubit_index] = np.sin(cur_rotation_angle)*temp_alpha + np.cos(cur_rotation_angle)*temp_beta
            

class Chromosome:
    """This class represents a solution of the Knapsack problem. 
    """
    def __init__(self, size : int, gene_size : int) -> None:
        self.chromosome_content : np.ndarray = np.empty(size, dtype=Gene)
        self.observed_chromosome : np.ndarray = np.zeros(size)
        for i in range(size):
            self.chromosome_content[i] = Gene.initialize_gene(gene_size)

    def update(self, b) -> None:
        gene : Gene
        for i, gene in enumerate(self.chromosome_content):
            gene.update_register(b.observed_chromosome[i], self.cur_profit, b.cur_profit)

    def make_observation(self) -> list:
        for i, gene in enumerate(self.chromosome_content):
            # For each gene, measure qubit
            self.observed_chromosome[i] = (gene.decode_gene())

    def evaluate_solution(self, profit_weights : np.ndarray) -> int:
        # Calculate total  profit of the solution
        self.cur_profit = np.sum(profit_weights * self.observed_chromosome)
        return self.cur_profit

    def repair_solution(self, cost_weights, cost_constraint) -> None:
        """This method repairs the Knapsack solutions so that they met the cost constraits.
        The algorithm is described in (Han & Kim, 2002)

        Parameters
        ----------
        cost_weights : np.Array
            List of weights for each position in the solution. Each position represents a specific item
        cost_constraint : int
            The constraint defined for the total cost
        """
        # Check if solution voilates the constraints
        cur_cost = np.sum(cost_weights * self.observed_chromosome)
        while cur_cost > cost_constraint:
            # remove random items until constraint is met
            cur_index = np.random.randint(len(self.chromosome_content))
            if self.observed_chromosome[cur_index] == 1:
                cur_cost = cur_cost - cost_weights[cur_index]
                self.observed_chromosome[cur_index] = 0

        while True:
            # Add items until the contstrain is voilated again
            cur_index = np.random.randint(len(self.chromosome_content))
            if self.observed_chromosome[cur_index] == 0:
                if cur_cost + cost_weights[cur_index] > cost_constraint:
                    break
                else:
                    cur_cost = cur_cost + cost_weights[cur_index]
                    self.observed_chromosome[cur_index] = 1


class Population:
    def __init__(self, population_size, chromosome_size, gene_size, cost_weights, cost_constraint, profit_weights, n_elite=3) -> None:
        self.n_elite = n_elite
        self.population : np.ndarray = np.empty(population_size, dtype=Chromosome)
        self.best_solutions = None
        self.cost_weights = cost_weights
        self.cost_constraint = cost_constraint
        self.profit_weights = profit_weights
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.gene_size = gene_size
        self.b = None

        for i in range(population_size):
            self.population[i] = (Chromosome(chromosome_size, gene_size))

    def global_migration(self, t, global_migration_period) -> None:
        if t % global_migration_period == 0:
            # Replace all solutions with the best solution (e.g repeat b to fill population)
            self.population = np.repeat(np.array([copy.deepcopy(self.b)]), self.population_size, axis=0)

    def local_migration(self):
        # The local migration is done by letting random pairs compete
        perm_list = np.random.permutation(len(self.best_solutions))
        for i in range(len(perm_list), 2):
            if self.best_solutions[i].cur_profit > self.best_solutions[i+1].cur_profit:
                # If the first solution has better fitness, replace the second solution
                self.best_solutions[i+1] = self.best_solutions[i]
            else:
                self.best_solutions[i] = self.best_solutions[i+1]

    def observe_population(self):
        for chromosome in self.population:
            chromosome.make_observation()
            chromosome.repair_solution(self.cost_weights, self.cost_constraint)
            chromosome.evaluate_solution(self.profit_weights)

        
        if not self.best_solutions:
            # If the B(t-1) list does not exsist, B(0) is created from scratch

            # Sort population based on fitness value. Note reverse for decending order since profit should be maximised
            self.population = sorted(self.population, key=attrgetter("cur_profit"), reverse=True)
            # Store best solutions
            self.b = copy.deepcopy(self.population[0])
            self.best_solutions = copy.deepcopy(self.population[:self.n_elite])
        else:
            # If in the while loop, combine the population with the previous best solutions
            self.best_solutions = sorted(np.concatenate((self.best_solutions, self.population)), key=attrgetter("cur_profit"), reverse=True)[:self.n_elite]
            # Check if the fitness value is better than previous best
            if self.b.cur_profit < self.best_solutions[0].cur_profit:
                print(self.best_solutions[0].cur_profit)
                self.b = copy.deepcopy(self.best_solutions[0])

        cur_chromosome : Chromosome
        for cur_chromosome in self.population:
            cur_chromosome.update(self.b)

    def add_to_plot(self, plot_array : np.ndarray, plot_counter : int) -> None:
        """This method plots a snapshot of the population
        """
        # We need to concatenate all genes into a single numpy array of two dimensions
        for i, chromo in enumerate(self.population):
            for j, gene in enumerate(chromo.chromosome_content):
                cur_gene_length = len(gene.q_register[0,:])
                for k in range(cur_gene_length):
                    plot_array[plot_counter, (i*2), j*cur_gene_length + k] = abs(gene.q_register[0, k])**2
                    plot_array[plot_counter, (i*2)+1, j*cur_gene_length + k] = abs(gene.q_register[1, k])**2
        

class QEA_main_knapsack:
    """Main routine for the QEA that solves the 0/1-Knapsack problem
    """
    def __init__(self, 
                 population_size=10, 
                 chromosome_size=100, 
                 gene_size=1, 
                 n_generations=1000, 
                 global_migration_phase=100, 
                 local_migration=False) -> None:
        self.n_generations = n_generations
        self.generation_count = 1 # t
        self.global_migration_phase = global_migration_phase
        self.local_migration = local_migration
        self.plot_array = np.zeros((10, 2*population_size*gene_size, chromosome_size))
        # Create cost and profit arrays
        np.random.seed(10)
        cost_weights = np.random.uniform(low=1, high=10, size=chromosome_size)
        np.random.seed()
        profit_weights = cost_weights + 5
        C = np.sum(cost_weights)/2
        print("cur_constraint: " + str(C))
        # Create the initial population
        self.population_obj = Population(population_size, chromosome_size, gene_size, cost_weights, C, profit_weights)
        self.population_obj.add_to_plot(self.plot_array, 0)
        self.population_obj.observe_population()

    def main(self):
        divider = self.n_generations // 10
        plot_counter = 1
        while self.generation_count < self.n_generations:
            # Print the first quantum chromosome
            if self.generation_count % divider == 0:
                self.population_obj.add_to_plot(self.plot_array, plot_counter)
                plot_counter += 1

            self.population_obj.observe_population()
            if self.global_migration_phase: self.population_obj.global_migration(self.generation_count, self.global_migration_phase)
            if self.local_migration: self.population_obj.local_migration()
            self.generation_count += 1


        fig, axs = plt.subplots(2, 5, figsize=(15, 15))
        for img_nr in range(plot_counter):
            axs[img_nr // 5][img_nr % 5].imshow(np.transpose(self.plot_array[img_nr, :, :]), vmin=0, vmax=1, cmap="hot")
            axs[img_nr // 5][img_nr % 5].set_title(f"Generation {img_nr * 10}")
            
        #plt.colorbar()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="hot"), cax=cbar_ax)
        plt.savefig(f"./knapsack_qga_logs/population_images/population_img.png")

        


if __name__=="__main__":
    # Create a test population
    """for test in range(10):
        freq_table = np.zeros(16)
        test_pop = Population(1000, 4, 1)
        test_pop.observe_population()
        #print(test_pop.population[0].observed_chromosome)
        for chromosome in test_pop.population:
            try:
                freq_table[int("".join(chromosome.observed_chromosome.astype("int").astype("str")),2)] += 1
            except:
                pdb.set_trace()

        cur_freq = pd.DataFrame(freq_table, columns=["freq"])
        plt.plot(freq_table)
    plt.show()
    """
    # Remove population images from previous run
    subprocess.run(["powershell", "-Command", "rm", "./knapsack_qga_logs/population_images/*"], shell=True)

    #np.random.seed(10)
    # Number of times the optimization process is repeated for different problem cases
    runs = 1
    best_results = np.zeros(runs)
    time_elapsed = np.zeros(runs)
    for i, run in enumerate(best_results):
        # For each run, optimize the Knapsack problem using the QEA
        start_time = time.time()
        test_qga = QEA_main_knapsack(
                    population_size=10,
                    chromosome_size=100,
                    gene_size=1,
                    n_generations=1000,
                    global_migration_phase=100,
                    local_migration=False
                )
        test_qga.main()
        # Store best result from the optimization
        best_results[i] = test_qga.population_obj.b.cur_profit
        # Store time elapsed
        time_elapsed[i] = time.time() - start_time

    print("Mean performance: " + str(np.mean(best_results)))
    print("Mean time elapsed: " + str(np.mean(time_elapsed)))
    