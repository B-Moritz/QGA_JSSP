## QGA_JSSP

### Content

The folder called "alg" contains different algorithms:
* Knapsack_qga.py - implementation of the QEA proposed by (Han & Kim, 2002). Run the file with python in an environment with numpy.

### Running the QMEA for JSSP
To run the algorithms, the python environment needs to be installed first. This can be done by creating a virtual environment locally on the computer and installing the content of the requirements.txt file. Note also that the install.py file must be executed to install the library code for the algorithm. The configuration for the algorithm execution can be changed in the QGA_JSSP\alg\NSGA_II\conf file location. To run the algorithm, pick a config file (for example experiment_qga_position.yaml) and run the following command:
python .\alg\NSGA_II\run_experiment.py -cn experiment_qga_position.yaml
 If docker is installed, the command docker compose build && docker compose up can be run in the root folder of the repository. 
The results of the algorithm run are stored under the logdata folder in the repository root. In that folder a csv file with the metrics for each repetition and iteration is saved as well as plots of the objective space and gnat-charts for some of the solutions in the pareto-front.



### References
Han, K.-H., & Kim, J.-H. (2002). Quantum-inspired evolutionary algorithm for a class of combinatorial optimization. IEEE Transactions on Evolutionary Computation, 6(6), 580–593. https://doi.org/10.1109/TEVC.2002.804320

