from scipy.sparse import coo_matrix
from juliacall import Main as jl
import time
import numpy as np
jl.seval('push!(LOAD_PATH, "pdhcg/julia_core/.")')
jl.seval('using PDHCG') 
jl.seval('using SparseArrays')
class QuadraticProgrammingProblem:
    def __init__(
                self, 
                objective_matrix, 
                objective_vector, 
                objective_constant, 
                constraint_matrix, 
                constraint_lower_bound, 
                num_equalities,
                variable_lower_bound, 
                variable_upper_bound,
                isfinite_variable_lower_bound,
                isfinite_variable_upper_bound
                 ):
        self.objective_matrix = coo_matrix(objective_matrix)
        self.objective_vector = objective_vector
        self.objective_constant = objective_constant
        self.constraint_matrix =  coo_matrix(constraint_matrix)
        self.constraint_lower_bound = constraint_lower_bound
        self.variable_lower_bound = variable_lower_bound
        self.isfinite_variable_lower_bound = isfinite_variable_lower_bound
        self.isfinite_variable_upper_bound = isfinite_variable_upper_bound
        self.variable_upper_bound = variable_upper_bound
        self.num_equalities = num_equalities
        self.n = objective_matrix.shape[0]
        self.m = constraint_matrix.shape[0]
        
def jl_to_scip(jl_matrix):
        row_indices, col_indices, values = jl.findnz(jl_matrix)
        return coo_matrix((np.array(values), (np.array(row_indices) - 1, np.array(col_indices) - 1)), shape = jl.size(jl_matrix))
    
def PyProblemFromJulia(qp, verbose = False) -> QuadraticProgrammingProblem:
    if verbose:
        print(f"Converting the problem to PyProblem")
        start_time = time.time()
    assert qp is not None, "The problem has not been set yet."
    objective_matrix = jl_to_scip(qp.objective_matrix)
    objective_vector = np.array(qp.objective_vector)
    objective_constant = qp.objective_constant
    constraint_matrix = jl_to_scip(qp.constraint_matrix)
    constraint_lower_bound = np.array(qp.right_hand_side)
    variable_lower_bound = np.array(qp.variable_lower_bound)
    variable_upper_bound = np.array(qp.variable_upper_bound)
    num_equalities = qp.num_equalities
    isfinite_variable_lower_bound = np.array(qp.isfinite_variable_lower_bound)
    isfinite_variable_upper_bound = np.array(qp.isfinite_variable_upper_bound)
    if verbose:
        end_time = time.time()  
        print("Time taken to convert the problem to PyProblem: ", end_time - start_time)
    return QuadraticProgrammingProblem(objective_matrix, objective_vector, objective_constant, constraint_matrix, 
                                       constraint_lower_bound, num_equalities, variable_lower_bound, variable_upper_bound, 
                                       isfinite_variable_lower_bound, isfinite_variable_upper_bound)

def read_problem(filename, verbose = False)->QuadraticProgrammingProblem:
    if verbose:
        print(f"Reading problem from {filename}")
        start_time = time.time()
    qp = jl.PDHCG.readFile(filename)
    qp_py = PyProblemFromJulia(qp)
    if verbose:
        end_time = time.time()
        print("Time taken to read the problem: ", end_time - start_time)
    return qp_py

def generate_problem(problem_type, n, density, seed, verbose = False)->QuadraticProgrammingProblem:
    if verbose:
        print(f"Generating problem of type {problem_type} with n = {n}, density = {density} and seed = {seed}")
        start_time = time.time()
    qp = jl.PDHCG.generateProblem(problem_type, n = n, density = density, seed = seed)
    qp_py = PyProblemFromJulia(qp)
    if verbose:
        end_time = time.time()
        print("Time taken to generate the problem: ", end_time - start_time)
    return qp_py