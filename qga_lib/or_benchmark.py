"""
This python file contains the code used to parse the OR library benchmarks for jssp found at 
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt

Usage:
Instantiate the BenchmarkCollection object. Specify url and set make_web_request to True if data should be collected from web
Deffault is to load the benchmkars from a json stored under the same file location as this file (jssp_benchmarks.json).

attribute problem_matrix contains the problem definition. At index 0 are the machine order for each job. At index 1 are the durations
Rows are jobs, columns are operations:
(2, n_jobs, n_operations)
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import re
import pdb

class BenchmarkCollection:
    """ This class is used to extract, store and visualize the OR library JSSP Benchmars."""

    def __init__(self, make_web_request: bool=False, url: str="https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt", file_path=""):
        self.url = url

        self.cache_path = os.path.join(os.path.dirname(__file__), "jssp_benchmarks.json")
        
        if make_web_request:
            # Collection raw data and parse the benchmarks
            self.collect_from_web()
            self.parse_benchmarks()
            self.save_json()
        else:
            if os.path.exists(self.cache_path):
                # If local cache file was found, load the data from local file
                self.load_json()
            else:
                raise FileNotFoundError()
            

    def save_json(self):
        """This method is used to save the benchmark dictionary to local disc"""
        for name, benchmark in self.benchmark_collection.items():
            # Convert object to dict
            benchmark["problem_matrix"] = benchmark["problem_matrix"].tolist()
        
        with open(self.cache_path, "w") as write_file:
            json.dump(self.benchmark_collection, write_file)

    def load_json(self):
        """This method is used to load the benchmarks from local cache"""
        with open(self.cache_path, "r") as cache_file:
            self.benchmark_collection = json.load(cache_file)

        for name, benchmark in self.benchmark_collection.items():
            # Convert object to dict
            benchmark["problem_matrix"] = np.asarray(benchmark["problem_matrix"])

    def collect_from_web(self):
        """ This function fetches the raw text data for the jssp benchmarks using the url 
        """
        # Make http request
        resp = requests.get(self.url)
        if resp.ok:
            # Extract text data
            text_data = resp.text
        else:
            raise Exception("Request responded with error message " + resp.status_code + ": " + resp.reason)

        self.raw_text = text_data

    def parse_benchmarks(self):
        """This method is used to parse the raw text file collected from web"""
        self.benchmark_collection = {}
        # splitting text at the lines containing + signs
        splitted_text = re.split("[\+]+", self.raw_text)

        for i in range(2, len(splitted_text), 2):
            # For each benchmark in the splitted text store the name and further parse the tech sequence
            cur_name = splitted_text[i].replace("\n", "").replace("instance", "").strip()

            if cur_name == "EOF":
                return

            cur_problem_str = splitted_text[i+1].strip().split("\n")
            # Extract the description
            cur_description = cur_problem_str[0].strip()
            
            m_j_def = re.split("\s+", cur_problem_str[1].strip())
            # Parse number of jobs
            cur_n_jobs = int(m_j_def[0])
            # Parse number of machines
            cur_n_machines = int(m_j_def[1])
            # Define the benchmark matrix of shape (2, j, m) where index 0 contains the machine order for each job and 1 contains the duration of the operation
            prob_matrix = np.empty((2, int(cur_n_jobs), int(cur_n_machines)), dtype=int)

            for job_seq in range(2, len(cur_problem_str)):
                # For each job, extract the string of machine and duration
                # Remove spaces by splitting the string
                cur_problem_list = re.split("\s+", cur_problem_str[job_seq].strip())
                for k in range(0, len(cur_problem_list), 2):
                    # Extract the mahcine number and duration and store it in benchmark matrix
                    if cur_problem_list[k] == "":
                        print("Error")
                        
                    prob_matrix[0][job_seq-2][int(k/2)] = int(cur_problem_list[k]) 
                    prob_matrix[1][job_seq-2][int(k/2)] = int(cur_problem_list[k+1])

            # Create benchmark object and add it to benchmark collection
            self.benchmark_collection[cur_name] = {
                    "name" : cur_name,
                    "description" : cur_description,
                    "problem_matrix" : prob_matrix, 
                    "n_jobs" : cur_n_jobs,
                    "n_machines" : cur_n_machines
                }
    
    def print_benchmarks(self):
        # Print all the benchmarks
        for name, benchmark in self.benchmark_collection.items():
            print("Name: " + name)
            print(benchmark["problem_matrix"])
            print("\n+++++++++++++++++++++++++++++\n\n")

if __name__=="__main__":
    test_benchmark_collection = BenchmarkCollection(make_web_request=True)
    test_benchmark_collection.print_benchmarks()

    
