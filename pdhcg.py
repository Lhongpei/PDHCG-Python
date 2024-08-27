from juliacall import Main as jl
import juliacall
from typing import Optional
import numpy as np
jl.seval('push!(LOAD_PATH, "src/.")')
jl.seval('using PDHCG')

class PDHCG:
    def __init__(
        self,
        gpu_flag: bool = False,
        warm_up_flag: bool = False,
        verbose_level: int = 2,
        time_limit: float = 3600.0,
        relat_error_tolerance: float = 1e-6,
        iteration_limit: int = 2**31 - 1,
        quadratic_problem = None,
        ruiz_rescaling_iters: int = 10,
        l2_norm_rescaling_flag: bool = False,
        pock_chambolle_alpha: float = 1.0,
        artificial_restart_threshold: float = 0.2,
        sufficient_reduction: float = 0.2,
        necessary_reduction: float = 0.8,
        primal_weight_update_smoothing: float = 0.2,
        save_flag: bool = False,
        saved_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        
        ):
        self.gpu_flag = gpu_flag
        self.warm_up_flag = warm_up_flag
        self.verbose_level = verbose_level
        self.time_limit = time_limit
        self.relat_error_tolerance = relat_error_tolerance
        self.iteration_limit = iteration_limit
        self.ruiz_rescaling_iters = ruiz_rescaling_iters
        self.l2_norm_rescaling_flag = l2_norm_rescaling_flag
        self.pock_chambolle_alpha = pock_chambolle_alpha
        self.artificial_restart_threshold = artificial_restart_threshold
        self.sufficient_reduction = sufficient_reduction
        self.necessary_reduction = necessary_reduction
        self.primal_weight_update_smoothing = primal_weight_update_smoothing
        self.save_flag = save_flag
        self.saved_name = saved_name
        self.output_dir = output_dir
        self.result = None
        self.solved = False
        self.qp = quadratic_problem
    def solve(
            self,
            warm_start_flag: bool = False,
            warm_start_primal = None,
            warm_start_dual = None
            ):
        if warm_start_flag:
            warm_start_primal = np.array(warm_start_primal, dtype = np.float64)
            warm_start_dual = np.array(warm_start_dual, dtype = np.float64)
        assert self.qp is not None, "The problem has not been set yet."
        self.result = jl.PDHCG.pdhcgSolve(self.qp, gpu_flag = self.gpu_flag, warm_up_flag = self.warm_up_flag, 
                                          verbose_level = self.verbose_level, time_limit = self.time_limit, 
                                          relat_error_tolerance = self.relat_error_tolerance, 
                                          iteration_limit = self.iteration_limit, ruiz_rescaling_iters = self.ruiz_rescaling_iters, 
                                          l2_norm_rescaling_flag = self.l2_norm_rescaling_flag, 
                                          pock_chambolle_alpha = self.pock_chambolle_alpha, 
                                          artificial_restart_threshold = self.artificial_restart_threshold, 
                                          sufficient_reduction = self.sufficient_reduction, 
                                          necessary_reduction = self.necessary_reduction, 
                                          primal_weight_update_smoothing = self.primal_weight_update_smoothing, 
                                          save_flag = self.save_flag, saved_name = self.saved_name, output_dir = self.output_dir,
                                          warm_start_flag = warm_start_flag, initial_primal = juliacall.convert(jl.Vector, warm_start_primal), 
                                          initial_dual = juliacall.convert(jl.Vector, warm_start_dual))
        self.solved = True
    def set_problem(self, qp):
        assert isinstance(qp, jl.PDHCG.QuadraticProgrammingProblem), "The problem should be an instance of PDHCG.QuadraticProgrammingProblem."
        self.qp = qp
        
    def read(self, filename, fixformat = False):
        self.qp = jl.PDHCG.readFile(filename, fixformat = fixformat)
        
    def setGeneratedProblem(self, problem_type, n, density, seed):
        self.qp = jl.PDHCG.generateProblem(problem_type, n = n, density = density, seed = seed)
        
    def setConstructedProblem(self, objective_matrix, objective_vector, objective_constant, constraint_matrix, 
                    constraint_lower_bound, variable_lower_bound = None, variable_upper_bound = None):
        self.qp = jl.PDHCG.QuadraticProgrammingProblem(objective_matrix, objective_vector, objective_constant, constraint_matrix, 
                                                        constraint_lower_bound, variable_lower_bound, variable_upper_bound)
    
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
    
    
if __name__ == '__main__':
    pdhcg = PDHCG()
    qp = jl.PDHCG.generateProblem("randomqp", n = 100, density = 0.1, seed = 0)
    pdhcg.solve(qp)