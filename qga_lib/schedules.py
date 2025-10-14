# This file containss library code for different schedule objects for optimizing the jssp problem
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import pandas as pd
import copy

from operations import Operation

from matplotlib import colormaps
import matplotlib as mpl

class Schedule:

    def __init__(self, operation_list: List[Operation], n_jobs: int, n_machines: int, jssp_problem: np.ndarray, objective_1: str, objective_2: str):
        # A schedule consists of a list of operations where each operation has a start time and a duration as, 
        # well as a machine and job assigned to itself
        self.operation_list: List[Operation] = operation_list
        #self.permutation = operation_list
        self.activated: bool = False
        self.n_machines: int = n_machines
        self.n_jobs: int = n_jobs
        self.jssp_problem: np.ndarray = jssp_problem
        # Determine the two objectives
        self.objective_dicts = {
                                "Makespan" : self.get_makespan, 
                                "Mean Flow Time" : self.get_mean_flow_time, 
                                "Mean Completion Time" : self.get_mean_completion_time
                                }
        if objective_1 not in self.objective_dicts.keys():
            raise Exception(f"The provided identifier provided as objective_1 was not recognised. Provided objective: {objective_1}")
        
        if objective_2 not in self.objective_dicts.keys():
            raise Exception(f"The provided identifier provided as objective_2 was not recognised. Provided objective: {objective_2}")
        
        self.objective_1 = self.objective_dicts[objective_1]
        self.objective_2 = self.objective_dicts[objective_2]

        
        

    def activate_schedule(self):
        # Performs the Hybrid gifflar and thompson algorithm proposed by (Varela et al., 2005)
        # Input is the operation sequence, the technical sequence and the duration matrix
        active_schedule = []
        # Making a copy of the semi activve schedule
        semi_active_schedule = copy.deepcopy(self.operation_list.tolist())
        # Initiating the timing lists for machine and job
        m_time = np.zeros(self.n_machines)
        j_time = np.zeros(self.n_jobs)
        T_counter = np.zeros(self.n_jobs, dtype=int)
        

        while len(semi_active_schedule) > 0:
            # Need to find the operation with the lowest completion time among the set of scheduable operations
            cur_schedulable_index = []
            cur_best_index_k = 0
            cur_lowest_complete_time = 0
            # Extract the start and comlpetion time for the first operation in the list
            #lowest_start_time_i  = np.max([m_time[semi_active_schedule[0].machine], j_time[semi_active_schedule[0].job]])
            cur_lowest_complete_time = np.inf#lowest_start_time_i + semi_active_schedule[0].duration

            # Loop to find the lowest completion time
            for i in range(0, len(semi_active_schedule)):
                # For each operation, find the lowest start time and the completion time
                cur_next_machine = self.jssp_problem[0][semi_active_schedule[i].job][T_counter[semi_active_schedule[i].job]]
                if cur_next_machine == semi_active_schedule[i].machine:
                    # If the current operation matches the machine, it is regarded as schedulable
                    cur_schedulable_index.append(i)
                    lowest_start_time_i  = np.max([m_time[semi_active_schedule[i].machine], j_time[semi_active_schedule[i].job]])
                    completion_i = lowest_start_time_i + semi_active_schedule[i].duration
                    
                    if completion_i < cur_lowest_complete_time:
                        # Update the lowest completion time if it is lower than the current lowest
                        cur_lowest_complete_time = completion_i
                        # store the index of the operation
                        cur_best_index_k = i

            # The lowest completion time
            found_lowest_completion_time = cur_lowest_complete_time
            # The machine number of the selected operation
            machine_k = semi_active_schedule[cur_best_index_k].machine

            overall_best_index = 0
            overall_lowest_start_time = np.inf #np.max([m_time[semi_active_schedule[overall_best_index].machine], j_time[semi_active_schedule[overall_best_index].job]])
            # The candidate set that contians the further reduced schedulable operations
            B = []
            for operation_index in cur_schedulable_index:
                # Find the operation with the lowest start time
                lowest_start_time_i  = np.max([m_time[semi_active_schedule[operation_index].machine], j_time[semi_active_schedule[operation_index].job]])
                if (semi_active_schedule[operation_index].machine == machine_k) and (lowest_start_time_i < found_lowest_completion_time):
                    # If the current operation is for machine_k and the start time is lower than the lowest completion time
                    # Regard it as part of the candidate operations
                    B.append(operation_index)
                    # Check if the start time 
                    #if  & (overall_lowest_start_time > lowest_start_time_i):
                        # If the starting time is lower than the the lowest completion time and the start time is lower than the previous lowest start time
                        # Store index and start time for that operation
                    #    overall_best_index = operation_index
                    #    overall_lowest_start_time = lowest_start_time_i
                    
            # Schedule the selected operation

            # Find the leftmose operation in the chromosome that is part of candidate set B
            selected_operation = semi_active_schedule.pop(np.min(B))
            selected_operation.start = np.max([m_time[selected_operation.machine], j_time[selected_operation.job]])
            # Update the job and machine next counters
            m_time[selected_operation.machine] = selected_operation.get_completion_time()
            j_time[selected_operation.job] = selected_operation.get_completion_time()
            # Add to the machine index counter
            T_counter[selected_operation.job] += 1
            active_schedule.append(selected_operation)
        
        self.activated = True
        self.operation_list = active_schedule

    def activate_schedule_old(self):
        # Performs the Hybrid gifflar and thompson algorithm proposed by (Varela et al., 2005)
        # Input is the operation sequence, the technical sequence and the duration matrix
        active_schedule = []
        # Making a copy of the semi activve schedule
        semi_active_schedule = copy.deepcopy(self.permutation.tolist())
        # Initiating the timing lists for machine and job
        m_time = np.zeros(self.n_machines)
        j_time = np.zeros(self.n_jobs)
        T_counter = np.zeros(self.n_jobs, dtype=int) 
        

        while len(semi_active_schedule) > 0:
            # Need to find the operation with the lowest completion time among the set of scheduable operations
            cur_schedulable_index = np.ones(self.n_jobs, dtype=int) * -1
            cur_best_index_k = 0
            cur_lowest_complete_time = 0
            # Extract the start and comlpetion time for the first operation in the list
            #lowest_start_time_i  = np.max([m_time[semi_active_schedule[0].machine], j_time[semi_active_schedule[0].job]])
            cur_lowest_complete_time = np.inf#lowest_start_time_i + semi_active_schedule[0].duration
            i = 0
            while np.any(cur_schedulable_index == -1) and (i < len(semi_active_schedule)):

                cur_job = semi_active_schedule[i]
                if cur_schedulable_index[cur_job] == -1: #cur_next_machine == semi_active_schedule[i].machine:
                    # If the current operation matches the machine, it is regarded as schedulable
                    cur_schedulable_index[cur_job] = i
                    cur_machine = self.jssp_problem[0][cur_job][T_counter[cur_job]]
                    lowest_start_time_i  = np.max([m_time[cur_machine], j_time[cur_job]])
                    completion_i = lowest_start_time_i + self.jssp_problem[1][cur_job][T_counter[cur_job]]
                    
                    if completion_i < cur_lowest_complete_time:
                        # Update the lowest completion time if it is lower than the current lowest
                        cur_lowest_complete_time = completion_i
                        # store the index of the operation
                        cur_best_index_k = i

                i += 1

            # The lowest completion time
            found_lowest_completion_time = cur_lowest_complete_time
            # The machine number of the selected operation
            machine_k = self.jssp_problem[0][semi_active_schedule[cur_best_index_k]][T_counter[semi_active_schedule[cur_best_index_k]]]
            # The candidate set that contians the further reduced schedulable operations
            B = []
            for operation_index in cur_schedulable_index:
                # Find the operation with the lowest start time
                cur_job = semi_active_schedule[operation_index]
                cur_machine = self.jssp_problem[0][cur_job][T_counter[cur_job]]
                lowest_start_time_i  = np.max([m_time[cur_machine], j_time[cur_job]])
                if (cur_machine == machine_k) and (lowest_start_time_i < found_lowest_completion_time):
                    # If the current operation is for machine_k and the start time is lower than the lowest completion time
                    # Regard it as part of the candidate operations
                    B.append(operation_index)
                    # Check if the start time 
                    #if  & (overall_lowest_start_time > lowest_start_time_i):
                        # If the starting time is lower than the the lowest completion time and the start time is lower than the previous lowest start time
                        # Store index and start time for that operation
                    #    overall_best_index = operation_index
                    #    overall_lowest_start_time = lowest_start_time_i
                    
            # Scedule the selected operation

            # Find the leftmose operation in the chromosome that is part of candidate set B
            selected_operation = np.min(B)
            cur_job = semi_active_schedule[selected_operation]
            cur_machine = self.jssp_problem[0][cur_job][T_counter[cur_job]]
            cur_duration = self.jssp_problem[1][cur_job][T_counter[cur_job]]
            selected_operation_start = np.max([m_time[cur_machine], j_time[cur_job]])
            cur_operation = Operation(cur_job, cur_machine, cur_duration, selected_operation_start)
            # Update the job and machine next counters
            m_time[cur_machine] = cur_operation.get_completion_time()
            j_time[cur_job] = cur_operation.get_completion_time()
            # Add to the machine index counter
            T_counter[cur_operation.job] += 1
            # Remove the operation from the permutation
            semi_active_schedule.pop(selected_operation)
            active_schedule.append(cur_operation)
        
        self.activated = True
        self.operation_list = active_schedule

    def get_mean_flow_time(self) -> float:
        """Calculate the mean completion time. This objective should be minimized 

        Returns
        -------
        _type_
            _description_
        """
        sum = 0
        for job in range(self.n_jobs):
            # For each job find the last operations
            start_time = 0
            start_time_found = False
            machine_counter = self.n_machines
            for cur_operation in self.operation_list:
                if job == cur_operation.job:
                    if not start_time_found:
                        start_time_found = True
                        start_time = cur_operation.start
                    
                    if machine_counter == 1:
                        sum += cur_operation.get_completion_time() - start_time
                        break

                    machine_counter -= 1
        
        self.mean_flow_time = sum/self.n_jobs
        return self.mean_flow_time
    
    def get_mean_completion_time(self) -> float:
        """Calculate the mean completion time. This objective should be minimized 

        Returns
        -------
        float
            the mean completion time
        """
        sum = 0
        for job in range(self.n_jobs):
            # For each job find the last operations
            for i in range(1, len(self.operation_list)+1):
                if job == self.operation_list[-i].job:
                    sum += self.operation_list[-i].get_completion_time()
                    break
        
        self.mean_completion_time = sum/self.n_jobs
        return self.mean_completion_time
    

    def get_makespan(self) -> int:
        # Find the highest completion time among the operations in the schedule
        cur_max_completion_time = self.operation_list[0].get_completion_time()
        for i in range(1, len(self.operation_list)):
            cur_completion_time = self.operation_list[i].get_completion_time()
            if cur_max_completion_time < cur_completion_time:
                cur_max_completion_time = cur_completion_time

        self.max_completion_time = cur_max_completion_time
        return self.max_completion_time

    def plot_gnatt_chart(self):
        # Plot the operation list as gnatt chart
        op_list = []
        for op in self.operation_list:
            op_list.append({"Task" : "M" + str(op.machine), "start_time" : op.start, "stop_time" : op.get_completion_time(), "operation" : "O_" + str(op.job) + "," + str(op.machine)})

        df = pd.DataFrame(op_list)
        fig, ax = plt.subplots(1,1, figsize=(50, 50))
        start_time = 0
        stop_time = df["stop_time"].max()
        n_rows = df["Task"].unique().shape[0]
        color_dict = {i : np.random.randint(0, 256, 3) for i in range(0, n_rows)}
        # Initiate all cells to white (255, 255, 255)
        image = np.ones((n_rows, int(stop_time), 3), dtype=int) * 255
        for operation in op_list:
            cur_job = int(operation["operation"][2])
            machine_number = int(operation["Task"][-1])
            image[machine_number][int(operation["start_time"]) : int(operation["stop_time"])][:] = color_dict[cur_job]

        # Create color patches
        handlers = []
        for i, color in color_dict.items():
            handlers.append(mpatches.Patch(color=color/255, label=f'$Job\;{i}$'))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(handles=handlers, loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_title("Schedule Gantt Chart")
        ax.set_yticks(np.arange(1, len(df["Task"].unique().tolist())+1))
        ax.set_yticklabels(df["Task"].unique().tolist())

        ax.set_xticks(np.arange(0, stop_time).tolist())

        ax.imshow(image)
        #plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
        plt.show()

    def get_image(self, height=400, width=1000):
        max_time = self.max_completion_time
        if width <= max_time:
            width = max_time

        cur_image = np.ones((height, int(width), 3))
        
        for operation in self.operation_list:
            cur_image[int(np.floor(operation.machine*height/self.n_machines)) : int((operation.machine + 1)*height/self.n_machines), 
                      int(operation.start*width/max_time) : int(operation.get_completion_time()*width/max_time), 
                      :] = plt.cm.gist_rainbow(operation.job/self.n_jobs)[:3]
        
        return cur_image

    def plot_gnatt_img(self, height=400, width=1000, display=True, axis=None):
        max_time = self.max_completion_time
        if width <= max_time:
            width = max_time

        cur_image = self.get_image(height)

        if axis == None:
            fig, axis = plt.subplots(1, 1)#, figsize=(10, 10))

        img1 = axis.imshow(cur_image)
        axis.set_xticks(np.linspace(-0.5, width, 5))
        axis.set_xticklabels(np.linspace(0, max_time+1, 5))

        axis.set_yticks(np.linspace(0, height, self.n_machines))
        axis.set_yticklabels([f"M{i+1}" for i in range(self.n_machines)])
        if display:
            plt.show()

        return img1
    
    def get_letter_code(index: int, alphabet_size: int, start_index: int) -> str:
        """This funciton is used to create the letter code for the job identifiers

        Parameters
        ----------
        index : int
            The corresponding integer value of the job identifier
        alphabet_size : int
            THe number of different letters in the alphabet
        start_index : int
            The offsett index for the char. For capital letters, this value should be 65 because ord('A') -> 65

        Returns
        -------
        str
            The letter code for the provided job number
        """
        res_list = []
        counter = 0
        while True:
            cur_position = alphabet_size**counter
            rest = index % alphabet_size**(counter+1)
            contribution = rest // cur_position
            res_list.append(contribution)
            if index // alphabet_size**(counter+1) == 0:
                break
            counter += 1

        res_list.reverse()
        letter_indexes = np.array(res_list) + start_index
        res_str = ""
        for letter in letter_indexes:
            res_str += chr(letter)
        
        return res_str
    
    def get_letter_code(self, index: int, alphabet_size: int, start_index: int) -> str:
        """This funciton is used to create the letter code for the job identifiers

        Parameters
        ----------
        index : int
            The corresponding integer value of the job identifier
        alphabet_size : int
            THe number of different letters in the alphabet
        start_index : int
            The offsett index for the char. For capital letters, this value should be 65 because ord('A') -> 65

        Returns
        -------
        str
            The letter code for the provided job number
        """
        res_list = []
        counter = 0
        while True:
            cur_position = alphabet_size**counter
            rest = index % alphabet_size**(counter+1)
            contribution = rest // cur_position
            res_list.append(contribution)
            if index // alphabet_size**(counter+1) == 0:
                break
            counter += 1

        res_list.reverse()
        letter_indexes = np.array(res_list) + start_index
        res_str = ""
        for letter in letter_indexes:
            res_str += chr(letter)
        
        return res_str
    
    def get_operation_code_list(self, alphabet_size=26, start_index=65) -> list:
        str_operation_list = []
        job_counters = np.ones(shape=self.n_jobs)
        for operation in self.operation_list:
            # For each operation in the schedule -> find the operation code
            cur_job_number = operation.job
            # Get the number of times the job already has been processed
            cur_count = job_counters[cur_job_number]
            # Increment the job count for later iterations
            job_counters[cur_job_number] += 1
            letter_code = self.get_letter_code(cur_job_number, alphabet_size, start_index)
            str_operation_list.append(f"{letter_code}{int(cur_count)}")

        return str_operation_list

    