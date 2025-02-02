import numpy as np
from pdhcg import PDHCG

solver = PDHCG(name="Example QP Solver")

# Set parameters
solver.setParams(gpu_flag=False, verbose_level=1)

# Define a custom problem
objective_matrix = np.eye(10)
objective_vector = np.ones(10)
constraint_matrix = np.random.randn(5, 10)
constraint_lower_bound = np.zeros(5)

solver.setConstructedProblem(
    objective_matrix=objective_matrix,
    objective_vector=objective_vector,
    objective_constant=0.0,
    constraint_matrix=constraint_matrix,
    constraint_lower_bound=constraint_lower_bound,
    num_equalities=2,
    variable_lower_bound=np.zeros(10),
    variable_upper_bound=np.ones(10) * 5,
    isfinite_variable_lower_bound=np.full(10, True),
    isfinite_variable_upper_bound=np.full(10, True)
)

# Solve the problem
solver.solve()

# Output results
print("Objective Value:", solver.objective_value)
print("Primal Solution:", solver.primal_solution)
print("Dual Solution:", solver.dual_solution)