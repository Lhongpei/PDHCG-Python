# from juliacall import Main as jl
import juliacall
from . import jl
from typing import Optional
import numpy as np
from pdhcg.params import Params
from pdhcg.probs import QuadraticProgrammingProblem, PyProblemFromJulia
from scipy.sparse import coo_matrix
# jl.seval('push!(LOAD_PATH, "pdhcg/julia_core/.")')
# jl.seval('using PDHCG')

class PDHCG:
    def __init__(
        self,
        name="",
        ) -> None:
        self.result = None
        self.solved = False
        self.__qp = None
        self.name = name
        self.set_init_solution = False
        self.init_params()
        
    def setParams(self, **kwargs):
        self.Params.setParams(**kwargs)
    
    def setParam(self, Param: str, value: str, verbose: bool = False):
        self.Params.setParam(Param, value, verbose)
        
    def init_params(self):
        self.Params = Params()
        
    def set_solution(self, primal, dual):
        self.set_init_solution = True
        self.init_primal = primal
        self.init_dual = dual
        
    def solve(self):
        if self.set_init_solution:
            warm_start_primal = np.array(self.init_primal, dtype = np.float64)
            warm_start_dual = np.array(self.init_dual, dtype = np.float64)
        assert self.__qp is not None, "The problem has not been set yet."
        initial_primal = None if not self.set_init_solution else juliacall.convert(jl.Vector, warm_start_primal)
        initial_dual = None if not self.set_init_solution else juliacall.convert(jl.Vector, warm_start_dual)
        self.result = jl.PDHCG.pdhcgSolve(self.__qp, gpu_flag = self.Params.gpu_flag, warm_up_flag = self.Params.warm_up_flag, 
                                          verbose_level = self.Params.verbose_level, time_limit = self.Params.time_limit, 
                                          relat_error_tolerance = self.Params.relat_error_tolerance, 
                                          iteration_limit = self.Params.iteration_limit, ruiz_rescaling_iters = self.Params.ruiz_rescaling_iters, 
                                          l2_norm_rescaling_flag = self.Params.l2_norm_rescaling_flag, 
                                          pock_chambolle_alpha = self.Params.pock_chambolle_alpha, 
                                          artificial_restart_threshold = self.Params.artificial_restart_threshold, 
                                          sufficient_reduction = self.Params.sufficient_reduction, 
                                          necessary_reduction = self.Params.necessary_reduction, 
                                          primal_weight_update_smoothing = self.Params.primal_weight_update_smoothing, 
                                          save_flag = self.Params.save_flag, saved_name = self.Params.saved_name, output_dir = self.Params.output_dir,
                                          warm_start_flag = self.set_init_solution, initial_primal = initial_primal, 
                                          initial_dual = initial_dual)
        self.solved = True
        
    def setProblem(self, qp:QuadraticProgrammingProblem):
        assert isinstance(qp, QuadraticProgrammingProblem), "The problem should be an instance of PDHCG.QuadraticProgrammingProblem."
        self.setConstructedProblem(qp.objective_matrix, qp.objective_vector, qp.objective_constant, qp.constraint_matrix,
                                      qp.constraint_lower_bound, qp.num_equalities, qp.variable_lower_bound, qp.variable_upper_bound,
                                      qp.isfinite_variable_lower_bound, qp.isfinite_variable_upper_bound)
        
    def read(self, filename, fixformat = False):
        self.__qp = jl.PDHCG.readFile(filename, fixformat = fixformat)
        
    def setGeneratedProblem(self, problem_type, n, density, seed):
        self.__qp = jl.PDHCG.generateProblem(problem_type, n = n, density = density, seed = seed)
        
    def setConstructedProblem(self, objective_matrix, objective_vector, objective_constant, constraint_matrix, 
                    constraint_lower_bound, num_equalities, variable_lower_bound, variable_upper_bound, 
                    isfinite_variable_lower_bound, isfinite_variable_upper_bound):
        obj_sparse = coo_matrix(objective_matrix)
        cons_sparse = coo_matrix(constraint_matrix)
        objective_matrix_jl = jl.sparse(
            jl.Vector(obj_sparse.row + 1),   
            jl.Vector(obj_sparse.col + 1),
            jl.Vector(obj_sparse.data),
            obj_sparse.shape[0],            
            obj_sparse.shape[1]             
        )
        constraint_matrix_jl = jl.sparse(
            jl.Vector(cons_sparse.row + 1),
            jl.Vector(cons_sparse.col + 1),
            jl.Vector(cons_sparse.data),
            cons_sparse.shape[0],
            cons_sparse.shape[1]
        )
        constraint_matrix_jl_t = jl.transpose(constraint_matrix_jl)
        m, n = constraint_matrix.shape
        self.__qp = jl.PDHCG.QuadraticProgrammingProblem(
            n, m,
            jl.Vector(variable_lower_bound),
            jl.Vector(variable_upper_bound),
            jl.Vector(isfinite_variable_lower_bound),
            jl.Vector(isfinite_variable_upper_bound),
            objective_matrix_jl,
            jl.Vector(objective_vector),
            objective_constant,
            constraint_matrix_jl,
            constraint_matrix_jl_t,
            jl.Vector(constraint_lower_bound),
            num_equalities
            )
        
    
    def returnProblem(self):
        return PyProblemFromJulia(self.__qp)
    @property
    def primal_solution(self):
        if self.solved == False:
            Warning("The problem has not been solved yet.")
        return list(self.result.primal_solution)
    
    @property
    def dual_solution(self):
        if self.solved == False:
            Warning("The problem has not been solved yet.")
        return list(self.result.dual_solution)
    
    @property
    def objective_value(self):
        if self.solved == False:
            Warning("The problem has not been solved yet.")        
        return self.result.objective_value
    
    @property
    def iteration_count(self):
        if self.solved == False:
            Warning("The problem has not been solved yet.")        
        return {"Outer-Iteration": self.result.iteration_count, "Inner(CG)-Iteration": self.result.CG_total_iteration}
    
    @property
    def solve_time_sec(self):
        if self.solved == False:
            Warning("The problem has not been solved yet.")        
        return self.result.solve_time_sec
    
    @property
    def kkt_error(self):
        if self.solved == False:
            Warning("The problem has not been solved yet.")        
        return self.result.kkt_error
    
    @property
    def status(self):
        if self.solved == False:
            Warning("The problem has not been solved yet.")        
        return self.result.termination_string