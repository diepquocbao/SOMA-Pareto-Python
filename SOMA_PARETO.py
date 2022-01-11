# --- SOMA Simple Program --- Version: SOMA PARETO (V1.0) August 25, 2020 -
# ------ Written by: Quoc Bao DIEP ---  Email: diepquocbao@gmail.com   ----
# -----------  See more details at the end of this file  ------------------
import numpy
import time
from List_of_CostFunctions import Schwefel as CostFunction

starttime = time.time()                                             # Start the timer
print('Hello! SOMA PARETO is working, please wait... ')
dimension = 10                                                      # Number of dimensions of the problem
# -------------- Control Parameters of SOMA -------------------------------
N_jump = 10                                                         # Assign values ​​to variables: Step, PRT, PathLength
PopSize, Max_Migration, Max_FEs = 100, 100, dimension*10**4         # Assign values ​​to variables: PopSize, Max_Migration
# -------------- The domain (search space) --------------------------------
VarMin, VarMax = -500, 500   # for Schwefel's function.                   # Define the search range
# %%%%%%%%%%%%%%      B E G I N    S O M A    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------- Create the initial Population -----------------------------
pop = VarMin + numpy.random.rand(dimension, PopSize) * (VarMax - VarMin) # Create the initial Population
fitness = CostFunction(pop)                                         # Evaluate the initial population
FEs = PopSize                                                       # Count the number of function evaluations
the_best_cost = min(fitness)                                        # Find the Global minimum fitness value
# ---------------- SOMA MIGRATIONS ----------------------------------------
C = numpy.around(PopSize * 0.04).astype(int)                        # Calculation of parameters A-C-D, please refer to the paper for more details
A = numpy.around(PopSize * 0.20).astype(int)                        # Calculation of parameters A-C-D, please refer to the paper for more details
D = numpy.around(PopSize * 0.16).astype(int)                        # Calculation of parameters A-C-D, please refer to the paper for more details
Migration = 0                                                       # Assign values ​​to variables: Migration
while FEs+N_jump <= Max_FEs:                                        # Terminate when reaching Max_FEs
    Migration = Migration + 1                                       # Increase Migration value
	# ------------ Control parameters -------------------------------------
    PRT = 0.50 + 0.45*numpy.cos(numpy.pi*FEs/Max_FEs+numpy.pi)      # Update PRT and Step parameters
    Step = 0.35 - 0.15*numpy.cos(numpy.pi*FEs/Max_FEs)              # Update PRT and Step parameters
    # ------------ Sort POP -----------------------------------------------
    pop_sort = numpy.append(fitness.reshape(1, PopSize),pop,axis=0) # Gather pop and fit into one
    pop_sort = pop_sort[:, pop_sort[0].argsort()]                   # Sort Pop according to the fitness values
    fitness = pop_sort[0,:]                                         # Split pop and fitness
    pop = pop_sort[1:,:]                                            # Split pop and fitness
    # ------------- Moving process ----------------------------------------
    Migrant_idx = numpy.random.choice(range(A,A+D),1)               # Migrant selection
    Migrant = pop[:, Migrant_idx].reshape(dimension, 1)             # Get the Migrant position (solution values) in the current population
    # ------------ Leader selection: k ------------------------------------
    Leader_idx = numpy.random.choice(range(C),1)                    # Leader selection
    Leader = pop[:, Leader_idx].reshape(dimension, 1)               # Get the Migrant position (solution values) in the current population
    offspring_path = numpy.empty([dimension, 0])                    # Create an empty path of offspring
    for move in range(N_jump):                                      # From Step to PathLength: jumping
        nstep     = (move+1) * Step
        PRTVector = (numpy.random.rand(dimension, 1) < PRT) * 1     # If rand() < PRT, PRTVector = 1, else, 0
        #PRTVector = (PRTVector - 1) * (1 - FEs/Max_FEs) + 1        # If rand() < PRT, PRTVector = 1, else, FEs/Max_FEs
        offspring = Migrant + (Leader - Migrant)*nstep*PRTVector    # Jumping towards the Leader
        offspring_path = numpy.append(offspring_path, offspring, axis=1) # Store the jumping path
    size = numpy.shape(offspring_path)                              # How many offspring in the path
    # ------------ Check and put individuals inside the search range if it's outside
    for cl in range(size[1]):                                       # From column
        for rw in range(dimension):                                 # From row: Check
            if offspring_path[rw][cl] < VarMin or offspring_path[rw][cl] > VarMax:  # if outside the search range
                offspring_path[rw][cl] = VarMin + numpy.random.rand() * (VarMax - VarMin) # Randomly put it inside
    # ------------ Evaluate the offspring and Update ----------------------
    new_cost = CostFunction(offspring_path)                         # Evaluate the offspring
    FEs = FEs + size[1]                                             # Count the number of function evaluations
    min_new_cost = min(new_cost)                                    # Find the minimum fitness value of new_cost
    # ----- Accepting: Place the best offspring into the current population
    if min_new_cost <= fitness[Migrant_idx]:                        # Compare min_new_cost with fitness value of the moving individual
        idz = numpy.argmin(new_cost)                                # Find the index of minimum value in the new_cost list
        fitness[Migrant_idx] = min_new_cost                         # Replace the moving individual fitness value
        pop[:, Migrant_idx[0]] = offspring_path[:, idz]             # Replace the moving individual position (solution values)
        # ----- Update the global best value --------------------
        if min_new_cost <= the_best_cost:                           # Compare Current minimum fitness with Global minimum fitness
            the_best_cost = min_new_cost                            # Update Global minimun fitness value
            the_best_value = offspring_path[:, idz]                 # Update Global minimun position
# %%%%%%%%%%%%%%%%%%    E N D    S O M A     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
endtime = time.time()                                               # Stop the timer
caltime = endtime - starttime                                       # Caculate the processing time
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Show the information to User
print('Stop at Migration :  ', Migration)
print('The number of FEs :  ', FEs)
print('Processing time   :  ', caltime, '(s)')
print('The best cost     :  ', the_best_cost)
print('Solution values   :  ', the_best_value)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This algorithm is programmed according to the descriptions in the papers 
# listed below:

# Link of paper: https://mendel-journal.org/index.php/mendel/article/view/87
# Diep Q. B., Zelinka I. and Das S. 2019. Self-Organizing Migrating Algorithm Pareto. MENDEL. 25, 1 (Jun. 2019), 111-120. DOI:https://doi.org/10.13164/mendel.2019.1.111.

# The control parameters PopSize, N_jump, and Step are closely related 
# and greatly affect the performance of the algorithm. Please refer to the 
# above paper to use the correct control parameters.