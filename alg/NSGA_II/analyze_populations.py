"""Script for producing insights on the population dumps prduced by the QMEA algorithm.
        Example usage:
        python analyze_populations.py 
            -lf "..\\..\\logdata_2\\Experiment_2_2025-01-30_22_07_18_122403" 
            -p "ft20" 
            -c "quantum_hash_base_encoding" 
            -i 0
"""


import pickle
import numpy as np
import pandas as pd
import os
import re
import pickle
import argparse

from nsga_population import *
from typing import List
from individual import Individual

import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable

#log_base = "..\\..\\logdata_2\\Experiment_2_2025-01-30_22_07_18_122403"
population_dump_location = "population_dumps"

def get_population(log_folder_path: str, dump_file_name: str) -> Population:
    """Function for loading population dump files

    Parameters
    ----------
    log_folder_path : str
        Path to folder in an experimentfolder that contains the population dumps
    dump_file_name : str
        The name of the pkl file that should be loaded

    Returns
    -------
    Population
        Retuns the population object that was written to disc during execution of the optimization algorithm
    """
    cur_path = os.path.join(log_folder_path, dump_file_name)
    with open(cur_path, "rb") as read_pop_dump:
        # Open pkl file and load the population
        cur_pop = pickle.load(read_pop_dump)

    return cur_pop

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="A script that produces insights of the population dumps")

    parser.add_argument("-lf", "--log_folder", type=str, required=True, help="The path to the logdata folder.")
    parser.add_argument("-p", "--problem_name", type=str, required=True, help="Name of benchmark problem (shorthand)")
    parser.add_argument("-c", "--candidate", type=str, required=True, help="Name of the algorithm executed.")
    parser.add_argument("-i", "--iteration", type=int, required=True, help="Iteration number.")

    parser.add_argument("-npe", "--n_permutation_imgs", type=int, default=3, help="The number of permutation images that should be created")
    parser.add_argument("-nb", "--n_bitstring_imgs", type=int, default=3, help="The number of bit string images that should be created")
    parser.add_argument("-npa", "--n_probab_imgs", type=int, default=3, help="The number of probability images that should be created")
    
    args = parser.parse_args()

    # Store arguments
    log_path = args.log_folder
    problem_name = args.problem_name
    candidate = args.candidate
    iteration = args.iteration
    # Number of plots to be created
    n_permutation_imgs = args.n_permutation_imgs
    n_bitstring_imgs = args.n_bitstring_imgs
    n_probab_imgs = args.n_probab_imgs
    # Check that the experiment folder exists
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Could not find {log_path}")
    
    print(f"Analyzing population in experiment {os.path.basename(log_path)}\n")

    # Obtain the list of file names of the population dumps in question
    base_path = os.path.join(log_path, population_dump_location)
    population_file_names = os.listdir(base_path)

    # Extract all dumps from the specified algorithm instance¨
    temp_filenames = []
    file_name_template = f"{problem_name}_{candidate}_{iteration}"
    for cur_dump_filename in population_file_names:
        if file_name_template in cur_dump_filename:
            temp_filenames.append(cur_dump_filename)

    if len(temp_filenames) == 0:
        # The specified log files were not found
        raise FileNotFoundError(f"The logfile with the signature {file_name_template} was not found. Please check the arguments provided to the script")

    n_generations = len(temp_filenames)
    # Sort the filenames according to generation number
    population_timeline = np.empty(n_generations, dtype=object) 
    for cur_dump_filename in temp_filenames:
        if f"{problem_name}_{candidate}_{iteration}" in cur_dump_filename:
            # Parse generation number from filename
            index_str = re.findall("_([0-9]+?)\.pkl", cur_dump_filename)[0]
            # Store filename in ordered array
            population_timeline[int(index_str)-1] = cur_dump_filename

    # Definition of the pandas dataframe columns
    df_dict = {"Iteration" : [], "Individual" : [], "Permutation" : [], "binary_value" : [], "map_value" : [], "makespan" : [], "mean flow time" : [], "µ_a" : [], "µ_b" : []}
    # Load the population at generation 0
    cur_pop = get_population(base_path, population_timeline[0])
    string_length = cur_pop.R[0].binary_chromosome.reshape(2, -1).shape[-1]
    # Initializing image arrays (amplitudes) - dimension: (n_generations, n_indiv_in_population, n_amplitudes, n_qubits_in_indiv)
    population_img = np.empty(shape=(len(population_timeline), len(cur_pop.R), 2, string_length), dtype=float)
    # Initializing image arrays (bit strings) - dimension: (n_generations, n_indiv_in_population, n_bits)
    population_img_x = np.empty(shape=(len(population_timeline), len(cur_pop.R), string_length), dtype=float)
    # Initializing image arrays (permutations after conversion) - dimension: (n_generations, n_indiv_in_population, n_integers_in_permutations)
    population_img_p = np.empty(shape=(len(population_timeline), len(cur_pop.R), cur_pop.R[0].permutation.shape[0]), dtype=float)

    for i, filename in enumerate(population_timeline):
        # For each generation, import population 
        cur_pop = get_population(base_path, filename)
        for ind_number, individual in enumerate(cur_pop.R):
            # For each individual in population extract the data
            
            # Calculating probabilities from amplitudes and reshaping array to two rows 
            cur_individual_string = (np.abs(individual.binary_chromosome)**2).reshape(2, -1)
            # Adding probabilities to the amplitude image array
            population_img[i, ind_number, :, :] = cur_individual_string
            # Adding the bit string of the individual to the bit string image
            population_img_x[i, ind_number, :] = individual.x.ravel().astype(int)
            # Adding the permutations to the permutation image
            population_img_p[i, ind_number, :] = individual.permutation
            # Calculate average probability for a or b
            mu_a, mu_b = cur_individual_string.mean(axis=1)
            # Add data to pandas dataframe
            df_dict["makespan"].append(individual.cur_fitness[0])
            df_dict["mean flow time"].append(individual.cur_fitness[1])
            df_dict["µ_a"].append(mu_a)
            df_dict["µ_b"].append(mu_b)

            if type(individual) == QChromosomeHashMultisetImprovedEncoding:
                df_dict["binary_value"].append(individual.convert_bin_to_decimal(individual.x))
                df_dict["map_value"].append(individual.uniform_mapping(individual.convert_bin_to_decimal(individual.x)))
            else:
                df_dict["binary_value"].append(None)
                df_dict["map_value"].append(None)

            df_dict["Permutation"].append(individual.permutation)
            df_dict["Individual"].append(ind_number)
            df_dict["Iteration"].append(i)
                
    # Create pandas dataframe with population information for each generation    
    population_df = pd.DataFrame(df_dict)
    population_df.to_csv(os.path.join(base_path, f"{problem_name}_{candidate}_dataset.csv"))
    # Normalized permutation numbers
    population_img_p = population_img_p/population_img_p.max()

    # --------------------
    # Plot permutation images for given generations
    generation_list = [0, n_generations-1]
    remaining_imgs = int(n_generations / (n_permutation_imgs - 1))
    # Find remaining generations for print
    for i in range(1, n_permutation_imgs - 1):
        generation_list.append(i*remaining_imgs)

    for p_img in generation_list:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title("Population permutations iter " + str(p_img))
        ax.imshow(population_img_p[p_img, :, :], cmap="inferno")
        fig.savefig(os.path.join(base_path, f"{problem_name}_{candidate}_perm_img_{p_img}.png"))
        plt.close(fig)
    # ---------------------
    # Plot permutation images for given generations
    generation_list = [0, n_generations-1]
    remaining_imgs = int(n_generations / (n_bitstring_imgs - 1))
    # Find remaining generations for print
    for i in range(1, n_bitstring_imgs - 1):
        generation_list.append(i*remaining_imgs)

    for x_img in generation_list:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title("Population permutations iter " + str(x_img))
        ax.imshow(population_img_x[x_img, :, :], cmap="inferno")
        fig.savefig(os.path.join(base_path, f"{problem_name}_{candidate}_x_img_{x_img}.png"))
        plt.close(fig)
    # ---------------------
    # Plot permutation images for given generations
    generation_list = [0, n_generations-1]
    remaining_imgs = int(n_generations / (n_probab_imgs - 1))
    # Find remaining generations for print
    for i in range(1, n_probab_imgs - 1):
        generation_list.append(i*remaining_imgs)

    for probab_img in generation_list:
        merged_img_a = population_img[:, :, 0, :].reshape(n_generations, -1, population_img.shape[-1])
        merged_img_b = population_img[:, :, 1, :].reshape(n_generations, -1, population_img.shape[-1])
        fig, ax = plt.subplots(2, 1, figsize=(20, 25))
        fig.suptitle("Populaiton at iteration " + str(probab_img))

        ax[0].set_title("µ_a")
        im1 = ax[0].imshow(merged_img_a[probab_img, :, :], cmap="inferno", vmax=1, vmin=0)

        ax[1].set_title("µ_b")
        im2 = ax[1].imshow(merged_img_b[probab_img, :, :], cmap="inferno", vmax=1, vmin=0)

        divider1 = make_axes_locatable(ax[0])
        cax1 = divider1.append_axes("bottom", size="5%", pad=0.5)
        divider2 = make_axes_locatable(ax[1])
        cax2 = divider2.append_axes("bottom", size="5%", pad=0.5)

        fig.colorbar(im1, label="Probability", orientation="horizontal", cax=cax1)
        fig.colorbar(im2, label="Probability", orientation="horizontal", cax=cax2)
        fig.savefig(os.path.join(base_path, f"{problem_name}_{candidate}_probab_img_{probab_img}.png"))
        plt.close(fig)
    # ---------------------
    if type(individual) == QChromosomeHashMultisetImprovedEncoding:
        # Plot distribution of binary values and map values
        fig, ax = plt.subplots(2, 1, figsize=(20, 25))
        fig.suptitle("Distribution of individual values before and after hash function")
        sns.histplot(data=population_df.reset_index(), x="binary_value", bins=10, ax=ax[0])
        sns.histplot(data=population_df.reset_index(), x="map_value", bins=10, ax=ax[1])
        fig.savefig(os.path.join(base_path, f"{problem_name}_{candidate}_distribution.png"))
        plt.close(fig)

if __name__=="__main__":
    main()
